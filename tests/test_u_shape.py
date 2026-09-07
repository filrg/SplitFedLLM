import copy
import pickle
import threading
import unittest
import weakref
from collections import defaultdict, deque
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.fine_tune.adapters import get_adapter
from src.fine_tune.u_shape import UShapeScheduler, FIELDS, ENVELOPE
from src.model.u_shape import build_partition, model_class, split_state_dict, join_state_dict, apply_lora, merge_lora


class Broker:
    """Thread-safe wire-level broker double with reordered/duplicated deliveries."""
    def __init__(self, reverse=False, duplicate=False):
        self.queues = defaultdict(deque)
        self.messages = []
        self.lock = threading.Lock()
        self.tag = 0
        self.unacked = set()
        self.reverse = reverse
        self.duplicate = duplicate

    def queue_declare(self, queue, durable=False):
        pass

    def confirm_delivery(self):
        pass

    def stop_consuming(self):
        pass

    def basic_publish(self, exchange, routing_key, body):
        with self.lock:
            self.messages.append((routing_key, pickle.loads(body)))
            self.queues[routing_key].append(body)
            if self.duplicate:
                self.queues[routing_key].append(body)

    def basic_get(self, queue, auto_ack):
        with self.lock:
            if not self.queues[queue]:
                return None, None, None
            self.tag += 1
            if not auto_ack:
                self.unacked.add(self.tag)
            pop = self.queues[queue].pop if self.reverse else self.queues[queue].popleft
            return SimpleNamespace(delivery_tag=self.tag), None, pop()

    def basic_ack(self, delivery_tag):
        with self.lock:
            self.unacked.remove(delivery_tag)


KWARGS = {
    "Bert": dict(vocab_size=16, hidden_size=8, num_attention_heads=2,
                 intermediate_size=16, dropout_prob=0),
    "GPT2": dict(vocab_size=16, n_embd=8, n_head=2, dropout=0),
    "Llama": dict(vocab_size=16, hidden_size=64, num_attention_heads=1,
                  num_key_value_heads=1, intermediate_size=64),
}


def batch(name, size=2):
    return dict(input_ids=torch.randint(0, 16, (size, 4)),
                labels=torch.randint(0, 4, (size,)) if name == "Bert" else torch.randint(0, 16, (size, 4)),
                attention_mask=torch.tensor([[1, 1, 1, 0]] * size))


def run_group(name, owners, worker, loaders, broker=None, credit=2, quota=3, training=True, session="test"):
    broker = broker or Broker()
    ids = list(owners)
    schedulers = {node: UShapeScheduler(name, node, broker, "cpu", session=session, worker_id="worker",
                  owner_ids=ids, max_inflight=credit, microbatches_per_window=quota,
                  worker_max_inflight=max(credit * len(ids), 1), timeout=10)
                  for node in (*ids, "worker")}
    results, errors = {}, []
    def execute(node):
        try:
            scheduler = schedulers[node]
            if node == "worker":
                results[node] = scheduler.worker(worker, 0.001, 0.01, training=training)
            else:
                results[node] = scheduler.owner(owners[node], loaders[node], 0.001, 0.01, training=training)
        except Exception as exc:
            errors.append(exc)
    threads = [threading.Thread(target=execute, args=(node,), daemon=True) for node in schedulers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    if any(thread.is_alive() for thread in threads):
        raise AssertionError("U-shape deadlock")
    if errors:
        raise errors[0]
    return schedulers, results, broker


class UShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def partitions(self, name, tail=1):
        full = model_class(name)(layer_id=0, n_block=3, **KWARGS[name])
        owner = build_partition(name, "owner", 1, 3, tail, **KWARGS[name])
        worker = build_partition(name, "worker", 1, 3, tail, **KWARGS[name])
        local, remote = split_state_dict(full.state_dict(), 1, 3, tail)
        owner.load_state_dict(local)
        worker.load_state_dict(remote)
        return full, owner, worker

    def test_all_models_match_monolithic_gradients_updates_and_wire_privacy(self):
        for name in KWARGS:
            with self.subTest(model=name):
                reference, owner, worker = self.partitions(name)
                batches = [batch(name, 1 if i == 4 else 2) for i in range(5)]
                schedulers, results, broker = run_group(name, {"alice": owner}, worker, {"alice": batches})
                optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0.01)
                adapter = get_adapter(name)
                for start in (0, 3):
                    optimizer.zero_grad()
                    window = batches[start:start + 3]
                    for data in window:
                        output, _ = adapter.forward(reference, data['input_ids'], data['attention_mask'])
                        (adapter.loss(output, data[adapter.label_key]) / len(window)).backward()
                    optimizer.step()
                actual = join_state_dict(owner.state_dict(), worker.state_dict(), 1, 3, 1)
                for key, value in reference.state_dict().items():
                    torch.testing.assert_close(actual[key], value, atol=2e-6, rtol=2e-5)
                actual_grads = join_state_dict(
                    {k: p.grad for k, p in owner.named_parameters()},
                    {k: p.grad for k, p in worker.named_parameters()}, 1, 3, 1)
                for key, param in reference.named_parameters():
                    torch.testing.assert_close(actual_grads[key], param.grad, atol=2e-6, rtol=2e-5)
                self.assertEqual(results, {'alice': (True, 9), 'worker': (True, 9)})
                self.assertEqual(schedulers['alice'].stats['front_forward'], 5)
                self.assertEqual(schedulers['alice'].stats['tail_forward'], 5)
                self.assertEqual(schedulers['worker'].stats['body_forward'], 5)
                self.assertEqual(schedulers['worker'].stats['steps'], 2)
                self.assertEqual(schedulers['alice'].stats['peak_pending'], 2)
                self.assertLessEqual(schedulers['worker'].stats['peak_pending'], 2)
                self.assertFalse(broker.unacked)
                for _, message in broker.messages:
                    self.assertEqual(set(message), ENVELOPE | FIELDS[message['type']])
                    self.assertFalse({'labels', 'label', 'input_ids', 'logits', 'prediction', 'loss'} & message.keys())
                    if 'tensor' in message:
                        self.assertTrue(message['tensor'].is_floating_point())
                        self.assertFalse(message['tensor'].requires_grad)
                        self.assertEqual(message['tensor'].shape[-1], KWARGS[name].get('hidden_size', 8))
                    if message.get('padding_mask') is not None:
                        self.assertEqual(message['padding_mask'].ndim, 2)

    def test_many_owners_uneven_empty_reordered_and_duplicate_delivery(self):
        _, template, worker = self.partitions('Bert')
        owners = {name: copy.deepcopy(template) for name in ('alice', 'bob', 'empty')}
        loaders = {'alice': [batch('Bert')] * 5, 'bob': [batch('Bert')] * 2, 'empty': []}
        schedulers, results, broker = run_group('Bert', owners, worker, loaders,
                                                Broker(reverse=True, duplicate=True))
        self.assertEqual(results['worker'], (True, 14))
        self.assertEqual(results['empty'], (True, 0))
        self.assertEqual(schedulers['worker'].stats['steps'], 2)
        self.assertEqual(schedulers['worker'].stats['body_forward'], 7)
        self.assertEqual(schedulers['worker'].stats['body_backward'], 7)
        self.assertEqual(schedulers['bob'].stats['steps'], 1)
        # Every COMMIT is preceded by all owners' drain notifications in that window.
        drained = defaultdict(set)
        for _, message in broker.messages:
            if message['type'] == 'DRAINED':
                drained[message['window']].add(message['owner'])
            if message['type'] == 'COMMIT':
                self.assertEqual(drained[message['window']], set(owners))

    def test_shared_worker_matches_group_objective(self):
        _, template, worker = self.partitions('Bert')
        owners = {name: copy.deepcopy(template) for name in ('alice', 'bob')}
        ref_owners, ref_worker = copy.deepcopy(owners), copy.deepcopy(worker)
        loaders = {'alice': [batch('Bert')] * 3, 'bob': [batch('Bert')]}
        run_group('Bert', owners, worker, loaders)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.01)
                      for m in (*ref_owners.values(), ref_worker)]
        for optimizer in optimizers:
            optimizer.zero_grad()
        adapter = get_adapter('Bert')
        for name, batches in loaders.items():
            for data in batches:
                a, _ = adapter.forward(ref_owners[name]['front'], data['input_ids'])
                h, _ = adapter.forward(ref_worker, a)
                logits, _ = adapter.forward(ref_owners[name]['tail'], h)
                (adapter.loss(logits, data['labels']) / 4).backward()
        for optimizer in optimizers:
            optimizer.step()
        for actual, expected in zip((*owners.values(), worker), (*ref_owners.values(), ref_worker)):
            for a, b in zip(actual.parameters(), expected.parameters()):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_evaluation_stays_local_and_never_updates_parameters(self):
        _, owner, worker = self.partitions('GPT2')
        before = copy.deepcopy(join_state_dict(owner.state_dict(), worker.state_dict(), 1, 3, 1))
        schedulers, results, broker = run_group('GPT2', {'alice': owner}, worker,
                                               {'alice': [batch('GPT2')] * 4}, training=False)
        after = join_state_dict(owner.state_dict(), worker.state_dict(), 1, 3, 1)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertEqual(schedulers['alice'].local_metrics['batches'], 4)
        self.assertEqual(schedulers['worker'].local_metrics['batches'], 0)
        self.assertFalse(any('BACKWARD' in msg['type'] for _, msg in broker.messages))
        self.assertEqual(schedulers['worker'].stats['steps'], 0)

    def test_checkpoint_roundtrip_with_zero_and_nonzero_tail(self):
        for name in KWARGS:
            for tail in (0, 1):
                with self.subTest(model=name, tail=tail):
                    full, owner, worker = self.partitions(name, tail)
                    reconstructed = join_state_dict(owner.state_dict(), worker.state_dict(), 1, 3, tail)
                    self.assertEqual(set(reconstructed), set(full.state_dict()))
                    self.assertFalse(any('lm_head' in k or 'classifier' in k for k in worker.state_dict()))
                    for key, value in full.state_dict().items():
                        torch.testing.assert_close(value, reconstructed[key], atol=0, rtol=0)

    def test_lora_training_and_merge_all_models(self):
        for name, tail in product(KWARGS, (0, 1)):
            with self.subTest(model=name, tail=tail):
                full, owner, worker = self.partitions(name, tail)
                owner = apply_lora(owner, name, 'owner', dict(r=2, alpha=4))
                worker = apply_lora(worker, name, 'worker', dict(r=2, alpha=4))
                run_group(name, {'alice': owner}, worker, {'alice': [batch(name)] * 2})
                merged_owner, merged_worker = merge_lora(owner, 'owner'), merge_lora(worker, 'worker')
                state = join_state_dict(merged_owner.state_dict(), merged_worker.state_dict(), 1, 3, tail)
                full.load_state_dict(state, strict=True)
                self.assertFalse(any('lora_' in key for key in state))

    def test_failure_aborts_without_committing(self):
        _, owner, worker = self.partitions('Bert')
        before = copy.deepcopy(worker.state_dict())
        data = batch('Bert')
        data['labels'].fill_(99)
        broker = Broker()
        with self.assertRaises(Exception):
            run_group('Bert', {'alice': owner}, worker, {'alice': [data]}, broker)
        self.assertFalse(any(m['type'] == 'COMMIT' for _, m in broker.messages))
        self.assertTrue(any(m['type'] == 'ABORT' for _, m in broker.messages))
        for key, value in worker.state_dict().items():
            torch.testing.assert_close(before[key], value, atol=0, rtol=0)

    def test_invalid_limits_and_timeout(self):
        for kwargs in ({'max_inflight': 0}, {'microbatches_per_window': -1}, {'worker_max_inflight': 0}):
            with self.assertRaises(ValueError):
                UShapeScheduler('Bert', 'alice', Broker(), 'cpu', session='s', worker_id='worker',
                                owner_ids=['alice'], **kwargs)
        scheduler = UShapeScheduler('Bert', 'alice', Broker(), 'cpu', session='s', worker_id='worker',
                                    owner_ids=['alice'], timeout=0.01)
        _, owner, _ = self.partitions('Bert')
        with self.assertRaises(TimeoutError):
            scheduler.owner(owner, [], 0.001)

    def test_dropout_uses_original_graph_and_releases_activations(self):
        with patch.dict(KWARGS, {'Bert': {**KWARGS['Bert'], 'dropout_prob': 0.3}}):
            reference, owner, worker = self.partitions('Bert')
        batches = [batch('Bert')] * 2
        refs = []
        hooks = [part.register_forward_hook(lambda m, args, out: refs.append(weakref.ref(out)))
                 for part in (owner['front'], worker, owner['tail'])]
        torch.manual_seed(123)
        run_group('Bert', {'alice': owner}, worker, {'alice': batches}, credit=1)
        for hook in hooks:
            hook.remove()
        self.assertEqual(len(refs), 6)
        self.assertTrue(all(ref() is None for ref in refs))
        torch.manual_seed(123)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0.01)
        optimizer.zero_grad()
        for data in batches:
            loss = get_adapter('Bert').loss(reference(data['input_ids']), data['labels'])
            (loss / 2).backward()
        optimizer.step()
        actual = join_state_dict(owner.state_dict(), worker.state_dict(), 1, 3, 1)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(actual[key], value, atol=2e-6, rtol=2e-5)

    def test_coordinator_abort_wakes_later_window(self):
        broker = Broker()
        scheduler = UShapeScheduler('Bert', 'alice', broker, 'cpu', session='s', worker_id='worker',
                                    owner_ids=['alice'])
        scheduler.window = 4
        message = dict(protocol=1, session='s', window=0, type='ABORT', owner='alice',
                       worker='worker', microbatch=-1)
        broker.basic_publish('', 'ushape_s_alice_ABORT', pickle.dumps(message))
        with self.assertRaisesRegex(RuntimeError, 'Peer aborted'):
            scheduler._poll('COMMIT')

    def test_rpc_and_coordinator_two_rounds_with_local_validation_and_fedavg(self):
        from src.RpcClient import RpcClient
        from src.UShapeServer import UShapeServer
        full, _, _ = self.partitions('Bert')
        coordinator = UShapeServer.__new__(UShapeServer)
        broker = Broker()
        coordinator.members = {'alice': 1, 'bob': 1, 'carol': 1, 'dave': 1,
                               'worker-a': 2, 'worker-b': 2}
        coordinator.u_config = {'max-inflight': 2, 'microbatches-per-window': 3,
                                 'worker-max-inflight': 4, 'timeout-seconds': 10}
        coordinator.initial = split_state_dict(full.state_dict(), 1, 3, 1)
        coordinator.model_name, coordinator.data_name = 'Bert', 'EMOTION'
        coordinator.cut_layers, coordinator.total_block, coordinator.tail_blocks = 1, 3, 1
        coordinator.batch_size, coordinator.num_sample, coordinator.control_count = 2, 4, 2
        coordinator.lr, coordinator.weight_decay, coordinator.clip_grad_norm = 0.001, 0.01, 0
        coordinator.fine_tune_config = {'enable': False, 'name': 'LoRA'}
        coordinator.validation, coordinator.save_parameters = True, False
        coordinator.round = 2
        coordinator.channel = coordinator.reply_channel = broker
        coordinator.logger = SimpleNamespace(log_info=lambda _: None)
        coordinator.send_to_response = lambda node, body: broker.basic_publish('', f'reply_{node}', body)
        clients = {node: RpcClient(node, role, broker, 'cpu') for node, role in coordinator.members.items()}
        data = [batch('Bert')] * 2
        real_builder = build_partition
        def small_builder(name, role, front, total, tail):
            return real_builder(name, role, front, total, tail, **KWARGS[name])
        coordinator._start_round()
        with patch('src.model.u_shape.build_partition', side_effect=small_builder), \
                patch('src.RpcClient.dataloader', return_value=data):
            for _ in range(2):
                errors = []
                def run_client(node):
                    try:
                        _, _, wire = broker.basic_get(f'reply_{node}', auto_ack=True)
                        clients[node].response_message(wire)
                    except Exception as exc:
                        errors.append(exc)
                threads = [threading.Thread(target=run_client, args=(node,), daemon=True) for node in clients]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(20)
                self.assertFalse(any(t.is_alive() for t in threads))
                if errors:
                    raise errors[0]
                for _ in clients:
                    method, _, wire = broker.basic_get('rpc_queue', auto_ack=False)
                    update = pickle.loads(wire)
                    self.assertEqual(update['action'], 'UPDATE')
                    self.assertNotIn('loss', update)
                    self.assertEqual(update['size'], 8 if update['layer_id'] == 2 else 4)
                    coordinator.on_request(broker, method, None, wire)
        self.assertEqual(coordinator.round, 0)
        for node in clients:
            _, _, wire = broker.basic_get(f'reply_{node}', auto_ack=True)
            self.assertEqual(pickle.loads(wire)['action'], 'STOP')
        merged = join_state_dict(*coordinator.initial, 1, 3, 1)
        full.load_state_dict(merged, strict=True)


if __name__ == '__main__':
    unittest.main()
