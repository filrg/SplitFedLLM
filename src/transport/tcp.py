"""Framed TCP tensors with pooled staging memory and asynchronous CUDA copies.

Socket threads never touch Pika. The scheduler owns autograd and the lifetime of
received GPU leases. Frames contain JSON metadata and raw contiguous tensor bytes.
"""
import collections
import hmac
import json
import queue
import socket
import struct
import threading
import time

import torch

from src.fine_tune.fair_queue import WeightedRoundRobin
from src.transport.pool import BufferPool

TENSOR_KINDS = {"BODY_FORWARD", "TAIL_FORWARD", "BODY_BACKWARD", "FRONT_BACKWARD"}
BASE = {"protocol", "session", "window", "type", "owner", "worker", "microbatch"}
DTYPES = {str(t).split(".")[-1]: t for t in
          (torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.uint8, torch.bool)}
HEADER_LIMIT = 16384


def _bytes(tensor):
    # Viewing storage as bytes also works for BF16, which NumPy cannot represent.
    return memoryview(tensor.view(torch.uint8).numpy()).cast("B")


def _read_into(sock, buffer):
    offset = 0
    while offset < len(buffer):
        received = sock.recv_into(buffer[offset:])
        if not received:
            raise EOFError("Truncated tensor frame")
        offset += received


def _read_json(sock):
    prefix = bytearray(4)
    first = sock.recv_into(prefix)
    if first == 0:
        return None
    if first < 4:
        _read_into(sock, memoryview(prefix)[first:])
    size = struct.unpack("!I", prefix)[0]
    if not 0 < size <= HEADER_LIMIT:
        raise ValueError("Invalid TCP frame header length")
    data = bytearray(size)
    _read_into(sock, memoryview(data))
    return json.loads(data)


def _write_json(sock, value):
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf8")
    if len(encoded) > HEADER_LIMIT:
        raise ValueError("TCP frame metadata too large")
    sock.sendall(struct.pack("!I", len(encoded)) + encoded)
    return len(encoded) + 4


class Received:
    def __init__(self, message, leases, ready, transport):
        self.message, self.leases, self.ready, self.transport = message, leases, ready, transport
        self.released = False

    def consume(self):
        if self.ready is not None:
            torch.cuda.current_stream(self.transport.device).wait_event(self.ready)
        return self.message

    def release(self):
        if not self.released:
            self.released = True
            event = None
            if self.ready is not None:
                # Also safe when a duplicate is discarded without consume().
                stream = torch.cuda.current_stream(self.transport.device)
                stream.wait_event(self.ready)
                consumed = torch.cuda.Event()
                consumed.record(stream)
                sent = torch.cuda.Event()
                sent.record(self.transport.copy_stream)
                # A transmitted view may alias a received buffer. Reuse must also
                # wait for D2H, without making the compute stream wait for it.
                with torch.cuda.stream(self.transport.reclaim_stream):
                    self.transport.reclaim_stream.wait_event(consumed)
                    self.transport.reclaim_stream.wait_event(sent)
                    event = torch.cuda.Event()
                    event.record(self.transport.reclaim_stream)
            for lease in self.leases:
                lease.release(event)
            self.leases.clear()
            with self.transport.lock:
                self.transport.receipts.discard(self)
            category = "gradient" if "BACKWARD" in self.message["type"] else "activation"
            self.transport.receive_slots[category].release()


class TcpTensorTransport:
    def __init__(self, node_id, device="cpu", bind_host="0.0.0.0", port=0,
                 pool_bytes=64 * 1024 * 1024, pinned="auto", queue_size=32, timeout=120):
        if pinned not in ("auto", "on", "off") or queue_size < 1 or timeout <= 0:
            raise ValueError("Invalid TCP transport configuration")
        self.node = str(node_id)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.timeout, self.queue_size = timeout, queue_size
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.base_session, self.secret, self.peers = None, None, {}
        self.error = None
        self.sequence = self.pending_sends = 0
        self.sockets, self.threads, self.senders = [], [], {}
        self.retired_sockets = set()
        self.inbox = collections.defaultdict(lambda: collections.defaultdict(collections.deque))
        self.receipts = set()
        self.fair = {}
        self.timing_events = []
        self.receive_slots = {kind: threading.BoundedSemaphore(queue_size) for kind in ("activation", "gradient")}
        self.metrics = dict(sent_bytes=0, received_bytes=0, sent_frames=0, received_frames=0,
                            d2h_ms=0.0, h2d_ms=0.0)
        use_pinned = self.device.type == "cuda" and pinned != "off"
        if use_pinned:
            try:
                torch.empty(1, pin_memory=True)
            except RuntimeError:
                if pinned == "on":
                    raise
                use_pinned = False
        self.pinned = use_pinned
        self.tx_pool = BufferPool(pool_bytes, pinned=use_pinned)
        # A live activation must never consume the gradient receive reserve.
        self.rx_pools = {kind: BufferPool(pool_bytes, pinned=use_pinned) for kind in ("activation", "gradient")}
        self.gpu_pools = {kind: BufferPool(pool_bytes, device=self.device) for kind in ("activation", "gradient")} if self.device.type == "cuda" else {}
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.receive_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.reclaim_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((bind_host, port))
        self.listener.listen(64)
        self.listener.settimeout(0.5)
        self.port = self.listener.getsockname()[1]
        self._thread(self._accept)

    def _thread(self, function, *args):
        thread = threading.Thread(target=function, args=args, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread

    def configure(self, session, secret, peers):
        self.drain()
        if self.base_session is not None:
            self._disconnect()
        if not secret or self.node not in peers:
            raise ValueError("TCP session requires a key and registered endpoints")
        self.base_session, self.secret = str(session), str(secret)
        self.peers = {str(k): dict(v) for k, v in peers.items()}
        self.fair.clear()

    def _fail(self, error):
        with self.lock:
            if not self.closed.is_set() and self.error is None:
                self.error = error

    def check(self):
        if self.error is not None:
            raise RuntimeError("TCP tensor transport failed; abort this round") from self.error
        if self.closed.is_set():
            raise RuntimeError("TCP tensor transport is closed")

    def _connect(self, peer):
        endpoint = self.peers[peer]
        sock = socket.create_connection((endpoint["host"], int(endpoint["port"])), self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout)
        with self.lock:
            self.sockets.append(sock)
        _write_json(sock, dict(node=self.node, session=self.base_session, key=self.secret))
        reply = bytearray(2)
        _read_into(sock, memoryview(reply))
        if reply != b"OK":
            raise ValueError("TCP session handshake rejected")
        return sock

    def _validate(self, message):
        kind = message.get("type")
        expected = BASE | {"tensor"} | ({"padding_mask"} if kind == "BODY_FORWARD" else set())
        if kind not in TENSOR_KINDS or set(message) != expected:
            raise ValueError("Invalid tensor message schema")
        if (message["protocol"] != 1 or message["session"] not in
                (self.base_session + "_train", self.base_session + "_eval")
                or message["owner"] not in self.peers or message["worker"] not in self.peers
                or type(message["window"]) is not int or message["window"] < 0
                or type(message["microbatch"]) is not int or message["microbatch"] < 0):
            raise ValueError("Invalid tensor session, route or version")

    def send(self, target, message):
        self.check()
        self._validate(message)
        if target not in self.peers:
            raise ValueError("Unregistered TCP peer")
        if sum(v.numel() * v.element_size() for v in message.values() if torch.is_tensor(v)) > self.tx_pool.limit:
            raise ValueError("Tensor frame exceeds send pool budget")
        host_leases, sources = [], []
        event = start = None
        try:
            specs, metadata = {}, {}
            for name, value in message.items():
                if torch.is_tensor(value):
                    lease = self.tx_pool.acquire(value.shape, value.dtype, timeout=self.timeout)
                    host_leases.append(lease)
                    if value.device.type == "cuda":
                        if value.device != self.device:
                            raise ValueError("Tensor belongs to another CUDA device")
                        source = value.detach().contiguous()
                        sources.append(source)  # Hold storage until D2H/send complete.
                        ready = torch.cuda.Event()
                        ready.record(torch.cuda.current_stream(self.device))
                        with torch.cuda.stream(self.copy_stream):
                            self.copy_stream.wait_event(ready)
                            if start is None:
                                start = torch.cuda.Event(enable_timing=True)
                                start.record(self.copy_stream)
                            lease.tensor.copy_(source, non_blocking=self.pinned)
                            source.record_stream(self.copy_stream)
                            event = torch.cuda.Event(enable_timing=True)
                            event.record(self.copy_stream)
                    else:
                        lease.tensor.copy_(value.detach())
                    specs[name] = dict(shape=list(value.shape), dtype=str(value.dtype).split(".")[-1],
                                       bytes=value.numel() * value.element_size())
                else:
                    metadata[name] = value
            # Independent TCP lanes avoid a forward waiting for activation memory
            # blocking the backwards that would free that very memory.
            lane = (target, "gradient" if "BACKWARD" in message["type"] else "activation")
            if lane not in self.senders:
                outgoing = queue.PriorityQueue(self.queue_size)
                self.senders[lane] = outgoing
                self._thread(self._sender, target, outgoing)
            with self.lock:
                self.sequence += 1
                sequence = self.sequence
                self.pending_sends += 1
            job = (metadata, specs, host_leases, sources, event, start)
            try:
                self.senders[lane].put((0 if "BACKWARD" in message["type"] else 1, sequence, job), timeout=self.timeout)
            except Exception:
                with self.lock:
                    self.pending_sends -= 1
                raise
        except Exception:
            for lease in host_leases:
                lease.release(event)
            raise

    def _sender(self, peer, outgoing):
        sock = None
        while not self.closed.is_set():
            try:
                _, _, job = outgoing.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                outgoing.task_done()
                return
            metadata, specs, leases, sources, event, start = job
            try:
                if event is not None:
                    event.synchronize()  # Only the network thread waits, not the GPU scheduler.
                if sock is None:
                    sock = self._connect(peer)
                size = _write_json(sock, dict(metadata=metadata, tensors=specs))
                for lease in leases:
                    view = _bytes(lease.tensor)
                    sock.sendall(view)
                    size += len(view)
                with self.lock:
                    self.metrics["sent_bytes"] += size
                    self.metrics["sent_frames"] += 1
                    if event is not None:
                        self.metrics["d2h_ms"] += start.elapsed_time(event)
            except Exception as exc:
                self._fail(exc)
            finally:
                for lease in leases:
                    lease.release(event)
                sources.clear()
                with self.lock:
                    self.pending_sends -= 1
                outgoing.task_done()

    def _accept(self):
        while not self.closed.is_set():
            try:
                sock, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            sock.settimeout(self.timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                self.sockets.append(sock)
            self._thread(self._reader, sock)

    def _reader(self, sock):
        authenticated = False
        try:
            hello = _read_json(sock)
            if not isinstance(hello, dict) or set(hello) != {"node", "session", "key"}:
                raise ValueError("Invalid TCP handshake")
            deadline = time.monotonic() + self.timeout
            while hello["session"] != self.base_session and not self.closed.is_set():
                if time.monotonic() >= deadline:
                    raise TimeoutError("TCP peer session not configured")
                time.sleep(0.01)
            if hello["node"] not in self.peers or not hmac.compare_digest(str(hello["key"]), self.secret):
                raise ValueError("Unregistered TCP connection")
            authenticated = True
            sock.sendall(b"OK")
            while not self.closed.is_set():
                frame = _read_json(sock)
                if frame is None:
                    return
                self._receive_frame(sock, hello["node"], frame)
        except (OSError, EOFError) as exc:
            if authenticated and not self.closed.is_set() and sock not in self.retired_sockets and sock.fileno() != -1:
                self._fail(exc)
        except Exception as exc:
            if authenticated:
                self._fail(exc)
        finally:
            sock.close()

    def _receive_frame(self, sock, peer, frame):
        if not isinstance(frame, dict) or set(frame) != {"metadata", "tensors"}:
            raise ValueError("Invalid TCP frame")
        metadata, specs = frame["metadata"], frame["tensors"]
        message = dict(metadata)
        if set(specs) - {"tensor", "padding_mask"} or "tensor" not in specs or set(specs) & set(metadata):
            raise ValueError("Invalid tensor fields")
        message.update({key: None for key in specs})
        self._validate(message)
        sender = message["owner"] if message["type"] in ("BODY_FORWARD", "BODY_BACKWARD") else message["worker"]
        recipient = message["worker"] if sender == message["owner"] else message["owner"]
        if sender != peer or recipient != self.node:
            raise ValueError("TCP connection does not match tensor route")
        category = "gradient" if "BACKWARD" in message["type"] else "activation"
        validated = []
        for name, spec in specs.items():
            if set(spec) != {"shape", "dtype", "bytes"} or spec["dtype"] not in DTYPES:
                raise ValueError("Invalid tensor descriptor")
            shape, dtype = spec["shape"], DTYPES[spec["dtype"]]
            if (not isinstance(shape, list) or not 1 <= len(shape) <= 8
                    or any(type(n) is not int or not 0 < n <= 2**30 for n in shape)):
                raise ValueError("Invalid tensor shape")
            numel = 1
            for dimension in shape:
                numel *= dimension
            if numel * torch.empty((), dtype=dtype).element_size() != spec["bytes"]:
                raise ValueError("Tensor byte count mismatch")
            if name == "tensor" and not dtype.is_floating_point:
                raise ValueError("Activations/gradients must be floating point")
            validated.append((name, shape, dtype, spec["bytes"]))
        # Check combined frame before acquiring leases; no partially allocated-frame deadlock.
        if sum(size for _, _, _, size in validated) > self.rx_pools[category].limit:
            raise ValueError("Tensor frame exceeds receive pool budget")
        slots = self.receive_slots[category]
        if not slots.acquire(timeout=self.timeout):
            raise TimeoutError("TCP receive queue is full")
        host, gpu, event = [], [], None
        try:
            total = 0
            for name, shape, dtype, size in validated:
                lease = self.rx_pools[category].acquire(shape, dtype, self.timeout)
                host.append(lease)
                _read_into(sock, _bytes(lease.tensor))
                if dtype.is_floating_point and not torch.isfinite(lease.tensor).all():
                    raise ValueError("Nonfinite tensor on wire")
                message[name] = lease.tensor
                total += size
            if self.device.type == "cuda":
                started = torch.cuda.Event(enable_timing=True)
                started.record(self.receive_stream)
                # Allocation is off the scheduler thread. Separate pools reserve gradient space.
                for name, shape, dtype, _ in validated:
                    lease = self.gpu_pools[category].acquire(shape, dtype, self.timeout)
                    gpu.append(lease)
                    with torch.cuda.stream(self.receive_stream):
                        lease.tensor.copy_(message[name], non_blocking=self.pinned)
                        event = torch.cuda.Event(enable_timing=True)
                        event.record(self.receive_stream)
                    message[name] = lease.tensor
                for lease in host:
                    lease.release(event)
                host.clear()
                with self.lock:
                    self.timing_events.append((started, event))
            receipt = Received(message, gpu or host, event, self)
            with self.lock:
                self.receipts.add(receipt)
                self.inbox[(message["session"], message["type"])][peer].append(receipt)
                self.metrics["received_bytes"] += total
                self.metrics["received_frames"] += 1
        except Exception:
            for lease in host + gpu:
                lease.release(event)
            slots.release()
            raise

    def poll(self, session, kind):
        self.check()
        self._collect_timings()
        key = (session, kind)
        with self.lock:
            incoming = self.inbox[key]
            eligible = [node for node, messages in incoming.items() if messages]
            if not eligible:
                return None
            if key not in self.fair:
                self.fair[key] = WeightedRoundRobin({node: peer.get("weight", 1) for node, peer in self.peers.items()})
            peer = self.fair[key].choose(eligible)
            receipt = incoming[peer].popleft()
        receipt.consume()
        return receipt

    def drain(self):
        deadline = time.monotonic() + self.timeout
        while self.pending_sends:
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError("Pending TCP sends did not complete")
            time.sleep(0.005)

    def _disconnect(self):
        for outgoing in self.senders.values():
            if self.closed.is_set():
                while True:
                    try:
                        _, _, job = outgoing.get_nowait()
                    except queue.Empty:
                        break
                    if job is not None:
                        for lease in job[2]:
                            lease.release(job[4])
                        with self.lock:
                            self.pending_sends -= 1
                    outgoing.task_done()
            else:
                outgoing.put((2, self.sequence + 1, None), timeout=self.timeout)
        self.senders.clear()
        with self.lock:
            sockets, self.sockets = self.sockets, []
            self.retired_sockets.update(sockets)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _collect_timings(self):
        with self.lock:
            waiting = []
            for start, end in self.timing_events:
                if end.query():
                    self.metrics["h2d_ms"] += start.elapsed_time(end)
                else:
                    waiting.append((start, end))
            self.timing_events = waiting

    def stats(self):
        self._collect_timings()
        return dict(self.metrics, pinned=self.pinned, pools={
            "tx_host": self.tx_pool.stats(),
            **{"rx_host_" + key: pool.stats() for key, pool in self.rx_pools.items()},
            **{"rx_gpu_" + key: pool.stats() for key, pool in self.gpu_pools.items()}})

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.listener.close()
        self._disconnect()
        for thread in self.threads:
            thread.join(timeout=2)
        for receipt in list(self.receipts):
            receipt.release()
        self.inbox.clear()
