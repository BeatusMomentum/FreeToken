"""Req.append_host writes into the preallocated buffer: value-equivalent to the
old per-step torch.cat, no reallocation, and existing views stay stable."""

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.prefill import ChunkedReq


def _mk(cls, input_ids, output_len=4):
    return cls(
        input_ids=input_ids,
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )


def test_append_host_matches_cat_without_reallocating():
    req = _mk(Req, torch.arange(6, dtype=torch.int32))
    ref = torch.arange(6, dtype=torch.int32)
    base_ptr = req.input_ids.data_ptr()
    view = req.input_ids[:3]
    for t in (101, 102, 103, 104):
        tok = torch.tensor([t], dtype=torch.int32)
        ref = torch.cat([ref, tok])
        req.append_host(tok)
        assert torch.equal(req.input_ids, ref)
        assert req.input_ids.data_ptr() == base_ptr
        assert req.input_ids.dtype == torch.int32
    assert len(req.input_ids) == req.max_device_len
    assert torch.equal(view, torch.arange(3, dtype=torch.int32))


def test_chunked_req_keeps_prompt_view_and_rejects_append():
    ids = torch.arange(6, dtype=torch.int32)
    req = _mk(ChunkedReq, ids)
    assert req.input_ids is ids
    with pytest.raises(NotImplementedError):
        req.append_host(torch.tensor([1], dtype=torch.int32))


@pytest.mark.parametrize("output_len", [1, 2, 128])
@pytest.mark.parametrize("overlap", [False, True])
def test_output_budget_counts_delivered_tokens(output_len, overlap):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from freetoken.core import Batch
    from freetoken.scheduler.scheduler import Scheduler

    req = _mk(Req, torch.arange(6, dtype=torch.int32), output_len)
    sent, freed = [], []
    running = {req}
    scheduler = SimpleNamespace(
        cache_manager=SimpleNamespace(lazy_free_region=nullcontext),
        finished_reqs=set(), eos_token_ids=set(), toolcall_anchor_id=None,
        decode_manager=SimpleNamespace(running_reqs=running, remove_req=running.discard),
        prefill_manager=SimpleNamespace(pending_list=[]), config=SimpleNamespace(page_size=1),
        status_reporter=SimpleNamespace(report_batch=lambda *args, **kw: None),
        _free_req_resources=freed.append, _kv_usage_pages=lambda: (0, 64),
        _mamba_slot_usage=lambda: None, _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0, send_result=sent.extend,
    )
    launched = 0
    for i in range(output_len):
        # A future GPU step can finish before the preceding token reaches the host.
        target = min(output_len, i + 1 + int(overlap))
        while launched < target:
            req.complete_one()
            launched += 1
        data = (
            SimpleNamespace(batch=Batch(reqs=[req], phase="decode")),
            (None, torch.tensor([100 + i], dtype=torch.int32), SimpleNamespace(synchronize=lambda: None)),
        )
        Scheduler._process_last_data(scheduler, data)
        assert len(sent) == i + 1
        assert sent[-1].finished == (i == output_len - 1)
    assert sent[-1].finish_reason == "length"
    assert req.input_ids.numel() == req.max_device_len
    assert freed == [req]
