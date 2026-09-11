"""Bounded buffer pools with CUDA-event deferred reuse."""
import threading
import time

import torch


class Lease:
    def __init__(self, pool, tensor):
        self.pool, self.tensor = pool, tensor
        self.released = False

    def release(self, event=None):
        if not self.released:
            self.released = True
            self.pool.release(self.tensor, event)


class BufferPool:
    def __init__(self, byte_limit, device="cpu", pinned=False):
        if type(byte_limit) is not int or byte_limit < 1:
            raise ValueError("Pool byte limit must be a positive integer")
        self.limit, self.device, self.pinned = byte_limit, torch.device(device), pinned
        self.lock = threading.Lock()
        self.free, self.pending = [], []
        self.allocated = self.in_use = self.peak = self.allocations = self.reuses = 0

    def _reap(self):
        waiting = []
        for tensor, event in self.pending:
            if event is None or event.query():
                self.free.append(tensor)
            else:
                waiting.append((tensor, event))
        self.pending = waiting

    def acquire(self, shape, dtype, timeout=5):
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * torch.empty((), dtype=dtype).element_size()
        if size > self.limit:
            raise ValueError("Tensor exceeds buffer pool limit; increase pool-bytes or reduce microbatch size")
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                self._reap()
                for index, tensor in enumerate(self.free):
                    if tuple(tensor.shape) == tuple(shape) and tensor.dtype == dtype:
                        self.free.pop(index)
                        self.in_use += size
                        self.reuses += 1
                        return Lease(self, tensor)
                while self.free and self.allocated + size > self.limit:
                    tensor = self.free.pop()
                    self.allocated -= tensor.numel() * tensor.element_size()
                    del tensor
                if self.allocated + size <= self.limit:
                    tensor = torch.empty(tuple(shape), dtype=dtype, device=self.device,
                                         pin_memory=self.pinned if self.device.type == "cpu" else False)
                    self.allocated += size
                    self.in_use += size
                    self.peak = max(self.peak, self.allocated)
                    self.allocations += 1
                    return Lease(self, tensor)
            if time.monotonic() >= deadline:
                raise TimeoutError("Buffer pool exhausted; graph credits/memory budget are incompatible")
            time.sleep(0.001)

    def release(self, tensor, event=None):
        with self.lock:
            self.in_use -= tensor.numel() * tensor.element_size()
            self.pending.append((tensor, event))
            self._reap()

    def stats(self):
        with self.lock:
            self._reap()
            return dict(allocated_bytes=self.allocated, leased_bytes=self.in_use, peak_bytes=self.peak,
                        allocations=self.allocations, reuses=self.reuses, pending_events=len(self.pending))
