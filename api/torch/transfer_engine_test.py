# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""E2E physical unit tests for TransferEngine on XLA TPUs."""

import threading
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from api.torch.transfer_engine import TransferEngine


class TransferEngineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize PyTorch XLA accelerator device E2E
    self.device = torch.device("tpu")
    self.num_layers = 1
    self.block_size = 1

  def test_initialization(self):
    device = torch.device("cpu")
    shape = (4, 128, 8)
    kv_caches = [torch.zeros(shape, device=device)]

    engine = TransferEngine(
        kv_caches=kv_caches,
        tp_rank=0,
        local_control_port=0,
        max_blocks=4,
        num_slots=2,
    )
    self.assertIsNotNone(engine)

  def test_e2e_transfer_polling(self):
    num_blocks = 2
    shape = (num_blocks, 128, 8)

    src_caches = []
    for _ in range(self.num_layers):
      src_caches.append(
          torch.full(
              shape, fill_value=1.0, dtype=torch.float32, device=self.device
          )
      )

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          torch.zeros(shape, dtype=torch.float32, device=self.device)
      )

    producer = TransferEngine(
        kv_caches=src_caches,
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    consumer = TransferEngine(
        kv_caches=dst_caches,
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    # Use getattr just in case local_control_port was completely hidden
    port = getattr(producer, "local_control_port", 0)
    self.assertGreater(port, 0)

    req_id = "test_req_poll"
    uuid = 12345
    producer.notify_for_read(req_id, uuid, [0, 1])

    remote_endpoint = f"127.0.0.1:{port}"
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=remote_endpoint,
        remote_block_ids=[0, 1],
        local_block_ids=[0, 1],
    )

    # Poll until consumer is done receiving
    done = False
    for _ in range(50):
      done_sending, done_recving, failed_recving = consumer.complete_read()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    # Check that consumer correctly loaded the values
    for t in dst_caches:
      np.testing.assert_allclose(t.cpu().numpy(), 1.0, atol=1e-5)

    # Poll producer until it's done sending
    done_prod = False
    for _ in range(50):
      done_sending, done_recving, failed_recving = producer.complete_read()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  def test_parallel_pull(self):
    num_blocks = 2
    shape = (num_blocks, 128, 8)

    src_caches = []
    for _ in range(self.num_layers):
      src_caches.append(
          torch.full(
              shape, fill_value=2.0, dtype=torch.float32, device=self.device
          )
      )

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          torch.zeros(shape, dtype=torch.float32, device=self.device)
      )

    producer = TransferEngine(
        kv_caches=src_caches,
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    consumer = TransferEngine(
        kv_caches=dst_caches,
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    port = getattr(producer, "local_control_port", 0)
    self.assertGreater(port, 0)

    req_id = "test_req_parallel"
    uuid = 99999
    producer.notify_for_read(req_id, uuid, [0, 1])

    remote_endpoint = f"127.0.0.1:{port}"
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=remote_endpoint,
        remote_block_ids=[0, 1],
        local_block_ids=[0, 1],
        parallelism=2,
    )

    done = False
    for _ in range(50):
      done_sending, done_recving, failed_recving = consumer.complete_read()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    for t in dst_caches:
      np.testing.assert_allclose(t.cpu().numpy(), 2.0, atol=1e-5)

    done_prod = False
    for _ in range(50):
      done_sending, done_recving, failed_recving = producer.complete_read()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  def test_e2e_segment_transfer_tp4_to_tp2(self):
    # Simulates mismatched TP4 (Prefill) -> TP2 (Decode) Serving Loop
    # Prefill node has 2 workers (ranks 0 and 1) pulling into 1 Decode rank (rank 0)
    # Prefill has size 1024 elements (4096 bytes), Decode has size 2048 elements (8192 bytes)
    # The sharded prefill tensors are sharded by double the TP world size of Decode
    p_shape = (1, 1024)
    d_shape = (1, 2048)

    # Initialize sharded prefill cache tensors with unique random values (not zeros)
    p0_cache = [torch.randn(p_shape, dtype=torch.float32, device=self.device).contiguous()]
    p1_cache = [torch.randn(p_shape, dtype=torch.float32, device=self.device).contiguous()]
    
    # Initialize decode cache tensor as zeros
    d0_cache = [torch.zeros(d_shape, dtype=torch.float32, device=self.device).contiguous()]

    # Force lazy tensor evaluation on TPU
    p0_cache[0].cpu()
    p1_cache[0].cpu()
    d0_cache[0].cpu()

    # Instantiate two prefill producers
    p0_engine = TransferEngine(
        kv_caches=p0_cache,
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    p1_engine = TransferEngine(
        kv_caches=p1_cache,
        tp_rank=1,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    # Instantiate one decode consumer
    d0_engine = TransferEngine(
        kv_caches=d0_cache,
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    p0_port = getattr(p0_engine, "local_control_port", 0)
    p1_port = getattr(p1_engine, "local_control_port", 0)
    self.assertGreater(p0_port, 0)
    self.assertGreater(p1_port, 0)

    req_id = "test_tp4_tp2_mismatch_transfer"
    uuid = 776655
    
    # Notify that both prefill ranks have block [0] ready
    p0_engine.notify_for_read(req_id, uuid, [0])
    p1_engine.notify_for_read(req_id, uuid, [0])

    # Decode rank 0 pulls segment 0 from prefill rank 0, and segment 1 from prefill rank 1
    # Segment 0: src block 0 (offset 0, size 4096 bytes) -> dst block 0 (offset 0)
    # Segment 1: src block 0 (offset 0, size 4096 bytes) -> dst block 0 (offset 4096)
    layers = [0]
    src_offsets = [0]
    
    dst_offsets_p0 = [0]
    dst_offsets_p1 = [4096]
    sizes = [4096]

    # Pull from prefill rank 0
    p0_endpoint = f"127.0.0.1:{p0_port}"
    fut_id0 = d0_engine.start_read_segments(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=p0_endpoint,
        local_block_ids=[0],
        layers=layers,
        src_offsets=src_offsets,
        dst_offsets=dst_offsets_p0,
        sizes=sizes,
        parallelism=1,
    )

    # Pull from prefill rank 1
    p1_endpoint = f"127.0.0.1:{p1_port}"
    fut_id1 = d0_engine.start_read_segments(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=p1_endpoint,
        local_block_ids=[0],
        layers=layers,
        src_offsets=src_offsets,
        dst_offsets=dst_offsets_p1,
        sizes=sizes,
        parallelism=1,
    )

    # Poll until consumer completes transfers
    done_p0 = False
    done_p1 = False
    for _ in range(50):
      done_sending, done_recving, failed_recving = d0_engine.complete_read()
      if req_id in failed_recving:
        self.fail("Segment transfer failed")
      if req_id in done_recving:
        # Check specific futures
        p0_done = d0_engine.wait_transfer(fut_id0)
        p1_done = d0_engine.wait_transfer(fut_id1)
        done_p0 = True
        done_p1 = True
        break
      time.sleep(0.05)

    self.assertTrue(done_p0 and done_p1, "Consumer did not finish segment transfers in time")

    # Fetch CPU copies of TPU tensors
    p0_numpy = p0_cache[0].cpu().numpy()
    p1_numpy = p1_cache[0].cpu().numpy()
    d0_numpy = d0_cache[0].cpu().numpy()

    # Assert that the exact random sharded values were transmitted successfully
    # First 1024 elements (4096 bytes) should equal prefill rank 0's cache
    np.testing.assert_allclose(d0_numpy[0, :1024], p0_numpy[0], atol=1e-5)
    # Next 1024 elements (4096 bytes) should equal prefill rank 1's cache
    np.testing.assert_allclose(d0_numpy[0, 1024:], p1_numpy[0], atol=1e-5)

    # Poll both producers until they are done sending
    done_prod0 = False
    done_prod1 = False
    for _ in range(50):
      done_send0, _, _ = p0_engine.complete_read()
      done_send1, _, _ = p1_engine.complete_read()
      if req_id in done_send0:
        done_prod0 = True
      if req_id in done_send1:
        done_prod1 = True
      if done_prod0 and done_prod1:
        break
      time.sleep(0.05)

    self.assertTrue(done_prod0, "Producer 0 did not finish sending in time")
    self.assertTrue(done_prod1, "Producer 1 did not finish sending in time")


if __name__ == "__main__":
  absltest.main()
