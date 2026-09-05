import copy
import pickle
import unittest
import weakref
from collections import defaultdict, deque
from unittest.mock import patch

import torch
from torch import nn

from src.fine_tune.adapters import TrainingAdapter, get_adapter, register_adapter
from src.fine_tune.scheduler import SplitScheduler, create_scheduler


class Channel:
    """In-memory RabbitMQ protocol double; no server or downloads required."""

    def __init__(self, first=False):
        self.queues = defaultdict(deque)
        self.published = []
        self.first = first
        self.max_pending = 0
        self.polls = 0

    def queue_declare(self, queue, durable=False):
        pass

    def basic_qos(self, prefetch_count):
        pass

    def basic_publish(self, exchange, routing_key, body):
        message = pickle.loads(body)
        self.published.append((routing_key, message))
        if self.first and routing_key == 'intermediate_queue_1':
            # d(sum(output**2))/d(output). Reverse response order to verify IDs.
            self.queues['gradient_queue_1_client'].appendleft({
                'data_id': message['data_id'], 'data': 2 * message['data'], 'trace': []})
            self.max_pending = max(self.max_pending, len(self.queues['gradient_queue_1_client']))
        if routing_key == 'rpc_queue':
            self.queues['reply_client'].append({'action': 'PAUSE'})

    def basic_get(self, queue, auto_ack):
        self.polls += 1
        if self.polls > 1000:
            raise AssertionError('Scheduler failed to terminate')
        if self.queues[queue]:
            return object(), None, pickle.dumps(self.queues[queue].popleft())
        return None, None, None


class FirstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.4), nn.Linear(5, 3))
        self.calls = 0

    def forward(self, input_ids):
        self.calls += 1
        return self.net(input_ids)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)

    def test_first_matches_original_graph_with_dropout_and_out_of_order_gradients(self):
        batches = [{'input_ids': torch.randn(2, 3), 'labels': torch.tensor([0, 1])}
                   for _ in range(5)]
        for window in (1, 2, 3, 10):
            with self.subTest(control_count=window):
                model = FirstModel()
                reference = copy.deepcopy(model)
                channel = Channel(first=True)
                trainer = create_scheduler('Bert', 'client', 1, channel, 'cpu')
                torch.manual_seed(123)
                result = trainer.first_layer(model, 0.01, 0.02, 0.7, window, batches)
                torch.manual_seed(123)
                optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.02)
                for start in range(0, len(batches), window):
                    optimizer.zero_grad()
                    losses = [reference(b['input_ids']).square().sum()
                              for b in batches[start:start + window]]
                    (sum(losses) / len(losses)).backward()
                    nn.utils.clip_grad_norm_(reference.parameters(), 0.7)
                    optimizer.step()
                self.assertEqual(result, (True, 5))
                self.assertEqual(model.calls, 5)  # Never forward on receipt of a gradient.
                self.assertLessEqual(channel.max_pending, window)
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected)
                self.assertTrue(all(t >= 0 for t in trainer.timings['comm']))
                self.assertEqual(len(trainer.timings['backward']), 5)

    def test_empty_loader_and_repeated_round_reset_counts(self):
        channel = Channel(first=True)
        trainer = create_scheduler('Bert', 'client', 1, channel, 'cpu')
        model = FirstModel()
        batch = {'input_ids': torch.randn(2, 3), 'labels': torch.tensor([0, 1])}
        for loader in ([batch], [batch], []):
            self.assertEqual(trainer.first_layer(model, 0.01, 0, 0, 1, loader),
                             (True, len(loader)))

    def test_original_outputs_are_released_after_backward(self):
        model = FirstModel()
        outputs = []
        handle = model.register_forward_hook(lambda module, args, output: outputs.append(weakref.ref(output)))
        trainer = create_scheduler('Bert', 'client', 1, Channel(first=True), 'cpu')
        batch = {'input_ids': torch.randn(2, 3), 'labels': torch.tensor([0, 1])}
        try:
            trainer.first_layer(model, 0.01, 0, 0, 2, [batch] * 3)
        finally:
            handle.remove()
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(ref() is None for ref in outputs))

    def test_invalid_window(self):
        trainer = create_scheduler('Bert', 'client', 1, Channel(), 'cpu')
        for window in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                trainer.first_layer(FirstModel(), 0.01, 0, 0, window, [])

    def test_unknown_gradient_fails_clearly(self):
        channel = Channel()
        channel.queues['gradient_queue_1_client'].appendleft({'data_id': 'unknown', 'data': []})
        trainer = create_scheduler('Bert', 'client', 1, channel, 'cpu')
        with self.assertRaisesRegex(ValueError, 'Unknown or duplicate'):
            trainer.first_layer(FirstModel(), 0.01, 0, 0, 1,
                                [{'input_ids': torch.randn(2, 3), 'labels': torch.tensor([0, 1])}])

    def test_registration_and_compatibility_entry_points(self):
        from src.fine_tune.Bert import Ft_Bert
        from src.fine_tune.GPT2 import Ft_GPT2
        from src.fine_tune.Llama import Ft_Llama
        for cls in (Ft_Bert, Ft_GPT2, Ft_Llama):
            self.assertIs(cls.first_layer, SplitScheduler.first_layer)
            self.assertIs(cls.last_layer, SplitScheduler.last_layer)
        register_adapter('custom_test', TrainingAdapter())
        self.assertIsInstance(create_scheduler('custom_test', 'c', 1, Channel(), 'cpu'), SplitScheduler)
        with self.assertRaisesRegex(ValueError, 'No training adapter'):
            get_adapter('missing')

    def test_adapter_losses_preserve_existing_objectives(self):
        logits = torch.randn(2, 4, 7)
        labels = torch.randint(0, 7, (2, 4))
        labels[0, -1] = 50256
        expected = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 7),
                                                labels[:, 1:].reshape(-1), ignore_index=50256)
        torch.testing.assert_close(get_adapter('GPT2').loss(logits, labels), expected)
        labels[0, -1] = -100
        torch.testing.assert_close(get_adapter('Llama').loss(logits, labels),
                                   nn.functional.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1)))

    def test_real_split_models_forward_once_and_last_layer_gradient(self):
        from src.model.Bert import Bert
        from src.model.GPT2 import GPT2
        from src.model.Llama import Llama
        factories = {
            'Bert': lambda layer: Bert(vocab_size=16, hidden_size=64,
                                       num_attention_heads=1, intermediate_size=128,
                                       dropout_prob=0, layer_id=layer, n_block=1),
            'GPT2': lambda layer: GPT2(vocab_size=16, n_head=1, n_embd=64,
                                       dropout=0, layer_id=layer, n_block=1),
            'Llama': lambda layer: Llama(vocab_size=16, hidden_size=64,
                                         num_attention_heads=1, num_key_value_heads=1,
                                         intermediate_size=128, layer_id=layer, n_block=1),
        }
        for name, factory in factories.items():
            with self.subTest(model=name):
                batch = {'input_ids': torch.randint(0, 16, (2, 4)),
                         'attention_mask': torch.ones(2, 4, dtype=torch.long),
                         'labels': torch.tensor([0, 1]) if name == 'Bert' else torch.randint(0, 16, (2, 4))}
                first = factory(1)
                channel = Channel(first=True)
                scheduler = create_scheduler(name, 'client', 1, channel, 'cpu')
                with patch.object(first, 'forward', wraps=first.forward) as forward:
                    self.assertEqual(scheduler.first_layer(first, 0.01, 0, 0, 2, [batch, batch, batch]), (True, 3))
                    self.assertEqual(forward.call_count, 3)
                messages = [m for q, m in channel.published if q == 'intermediate_queue_1']
                last = factory(2)
                reference = copy.deepcopy(last)
                last_channel = Channel()
                last_channel.queues['intermediate_queue_1'].extend(messages)
                last_channel.queues['reply_last'].append({'action': 'PAUSE'})
                last_scheduler = create_scheduler(name, 'last', 2, last_channel, 'cpu')
                self.assertEqual(last_scheduler.last_layer(last, 0.01, 0, 0), (True, 3))
                optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0)
                gradients = [m for q, m in last_channel.published if q == 'gradient_queue_1_client']
                for message, received in zip(messages, gradients):
                    optimizer.zero_grad()
                    inputs = torch.tensor(message['data'], requires_grad=True)
                    output, _ = get_adapter(name).forward(reference, inputs, message.get('attention_mask'))
                    get_adapter(name).loss(output, message['label']).backward()
                    torch.testing.assert_close(torch.tensor(received['data']), inputs.grad)
                    self.assertEqual(received['data_id'], message['data_id'])
                    self.assertEqual(received['trace'], [])
                    optimizer.step()
                for actual, expected in zip(last.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
