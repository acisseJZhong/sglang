"""Full prefill CUDA graphs of hybrid linear attention models (issue #48593).

The prefill runner captures one graph per token bucket over a fixed request axis
(full_prefill_max_req rows: the batch's requests, then zero-length sentinels) and
refreshes the attention metadata out of graph before each replay. The linear
backend's extend metadata must live in buffers that capture and replay share, sized
by that request axis.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import hybrid_linear_attn_backend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_REQ_SLOTS = 4
_PAD_SLOT_ID = -1


class _ForwardMetadataBackend(MambaAttnBackendBase):
    """A linear backend whose extend reads only the forward metadata."""

    captures_extend_on_forward_metadata = True


class _TablesBackend(MambaAttnBackendBase):
    """A linear backend whose extend runs on per-bucket prefill graph tables."""

    prefill_graph_max_seqs = 8

    def supports_prefill_graph_extend(self) -> bool:
        return True

    def init_prefill_graph_metadata(self, forward_batch):
        tables = SimpleNamespace(rows=[])
        self.refresh_prefill_graph_metadata(tables, forward_batch)
        return tables

    def refresh_prefill_graph_metadata(self, meta, forward_batch):
        meta.rows.append(forward_batch.req_pool_indices.tolist())
        self.forward_metadata = SimpleNamespace(mamba_track_indices=None)


def _backend(cls, *, extra_buffer=False):
    backend = cls.__new__(cls)
    backend.device = torch.device("cpu")
    # Request row r holds state slot 10 + r; row 0 maps to the padding slot 0.
    mapping = torch.arange(16, dtype=torch.int32) + 10
    mapping[0] = 0
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda rows: mapping[rows]
    )
    backend._state_ids = SimpleNamespace(ids=lambda ids: ids)
    # As __init__ leaves them; the decode graph runner sizes state_indices_list.
    backend.forward_metadata = None
    backend.state_indices_list = []
    backend._fused_state_indices_ok = False
    backend.replayssm_write_pos_list = None
    exec_ctx = SimpleNamespace(
        mamba=SimpleNamespace(enable_mamba_extra_buffer=extra_buffer)
    )
    with patch.object(hybrid_linear_attn_backend, "get_exec", return_value=exec_ctx):
        backend.init_prefill_cuda_graph_state(
            max_num_tokens=64, max_req_slots=_REQ_SLOTS
        )
    return backend


def _padded_batch(*, rows, lens, num_tokens):
    """A batch as the prefill runner hands it to the backend: the request axis
    padded to the slot count with zero-length rows (request row 0 at replay); the
    host lengths cover the real requests only at replay."""
    padded_lens = lens + [0] * (_REQ_SLOTS - len(lens))
    padded_rows = rows + [0] * (_REQ_SLOTS - len(rows))
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=_REQ_SLOTS,
        req_pool_indices=torch.tensor(padded_rows),
        extend_seq_lens=torch.tensor(padded_lens),
        extend_seq_lens_cpu=list(lens),
        extend_prefix_lens=torch.zeros(_REQ_SLOTS, dtype=torch.int64),
        input_ids=torch.zeros(num_tokens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(padded_lens),
        num_padding=0,
        spec_info=None,
        ragged_verify_layout=None,
        mamba_track_indices=None,
    )


class TestFullPrefillGraphMetadata(CustomTestCase):
    def test_replay_refreshes_the_captured_buffers(self):
        """Capture and every replay hand the kernels the same state-slot and
        query-offset tensors, refreshed in place for the batch: sentinel rows are
        empty sequences on PAD_SLOT_ID, and the bucket's padding tokens belong to
        no sequence. Before the fix, capture raised IndexError: the extend metadata
        came from the decode graph buffers, which the prefill runner never sizes."""
        backend = _backend(_ForwardMetadataBackend)
        backend.init_forward_metadata_out_graph(_capture_batch(64), in_capture=True)
        captured = backend.forward_metadata
        self.assertEqual(captured.query_start_loc.tolist(), [0, 64, 64, 64, 64])
        self.assertEqual(
            captured.mamba_cache_indices.tolist(), [0] + [_PAD_SLOT_ID] * 3
        )

        replay = _padded_batch(rows=[7, 3], lens=[5, 2], num_tokens=64)
        backend.init_forward_metadata_out_graph(replay)
        metadata = backend.forward_metadata
        self.assertIs(metadata.query_start_loc, captured.query_start_loc)
        self.assertIs(metadata.mamba_cache_indices, captured.mamba_cache_indices)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 5, 7, 7, 7])
        self.assertEqual(
            metadata.mamba_cache_indices.tolist(),
            [17, 13, _PAD_SLOT_ID, _PAD_SLOT_ID],
        )

    def test_tables_are_captured_per_bucket_over_the_real_requests(self):
        """Backends with prefill graph tables capture one per bucket and refresh
        that bucket's at replay, over the batch's requests without the sentinels."""
        backend = _backend(_TablesBackend)
        for num_tokens in (32, 64):
            backend.init_forward_metadata_out_graph(
                _capture_batch(num_tokens), in_capture=True
            )
        backend.init_forward_metadata_out_graph(
            _padded_batch(rows=[9], lens=[40], num_tokens=64)
        )
        tables = backend._full_prefill_graph.tables
        self.assertEqual(sorted(tables), [32, 64])
        self.assertEqual(tables[32].rows, [[0]])
        self.assertEqual(tables[64].rows, [[0], [9]])

    def test_unsupported_setups_raise_at_graph_state_init(self):
        """A backend that cannot capture its extend, or a forward-metadata backend
        with the extra_buffer prefix cache (whose track plan is host-built), fails
        before capture with an actionable error instead of an IndexError."""
        with self.assertRaisesRegex(ValueError, "breakable"):
            _backend(MambaAttnBackendBase)
        with self.assertRaisesRegex(ValueError, "extra_buffer"):
            _backend(_ForwardMetadataBackend, extra_buffer=True)


def _capture_batch(num_tokens):
    """The runner's capture batch: one dummy request of the whole bucket on request
    row 0, then sentinel rows, with host lengths for every row."""
    batch = _padded_batch(rows=[0], lens=[num_tokens], num_tokens=num_tokens)
    batch.req_pool_indices = torch.arange(_REQ_SLOTS)
    batch.extend_seq_lens_cpu = [num_tokens] + [0] * (_REQ_SLOTS - 1)
    return batch


if __name__ == "__main__":
    unittest.main()
