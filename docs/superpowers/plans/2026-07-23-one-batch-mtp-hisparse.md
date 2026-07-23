# One-Batch MTP + HiSparse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one-batch EAGLE MTP decode with target KV managed by HiSparse and draft KV retained in a dense HBM pool.

**Architecture:** The target runner keeps the existing HiSparse allocator, host pool, staging, and hydration lifecycle. The draft runner shares logical token locations but disables its own HiSparse coordinator and allocates a dense draft KV pool sized to the target logical capacity. HiSparse allocator backup/restore becomes a real transaction for EAGLE candidate allocations, and one-batch admission checks both target and draft capacity.

**Tech Stack:** Python, PyTorch, SGLang EAGLE V1, HiSparse, `unittest`.

## Global Constraints

- Do not change online scheduler behavior.
- Do not stage draft KV to host.
- Keep external draft models, correctness mode, and eager MTP profiling unsupported.
- Preserve exact output-length and MTP metric semantics.
- Only target KV participates in HiSparse staging and hydration.

---

### Task 1: Combination Validation And Admission Accounting

**Files:**
- Modify: `python/sglang/bench_one_batch_utils.py`
- Modify: `python/sglang/bench_one_batch.py`
- Test: `test/registered/unit/test_bench_one_batch_utils.py`

**Interfaces:**
- Produces: `DraftCapacitySnapshot(total: int, available: int)`.
- Extends: `evaluate_hisparse_wave_admission(..., speculative_reserve_per_request=0, draft_snapshot=None)`.

- [ ] **Step 1: Write failing helper tests**

Add a validation test that `enable_hisparse=True` is accepted for EAGLE, while correctness, eager profile, and external draft model remain rejected. Add admission tests where target capacity is sufficient but dense draft capacity rejects the next wave with `draft_device_pool`.

- [ ] **Step 2: Run the helper suite and verify RED**

Run: `PYTHONPATH=python python3 test/registered/unit/test_bench_one_batch_utils.py`

Expected: validation still raises `one-batch MTP does not yet support HiSparse`, and the new draft-capacity arguments are rejected.

- [ ] **Step 3: Implement minimal helper behavior**

Remove only the HiSparse rejection. Add immutable draft capacity and requirement fields. Compute draft required tokens as:

```python
draft_required = _align_up(
    (ready_count + 1) * (input_len + output_len), page_size
) + (ready_count + 1) * speculative_reserve_per_request
```

Reject with `draft_device_pool` when the dense draft pool cannot cover it.

- [ ] **Step 4: Pass draft capacity from one-batch**

Read total draft slots from `generation_worker.draft_model_runner.token_to_kv_pool.size`; derive available capacity from committed logical usage and speculative reserve. Add the values to capacity snapshots and stop-reason formatting.

- [ ] **Step 5: Run tests and commit**

Run: `PYTHONPATH=python python3 test/registered/unit/test_bench_one_batch_utils.py`

Expected: all tests pass.

Commit: `git commit -am "bench: account for draft KV in HiSparse MTP admission"`

### Task 2: Target-Only HiSparse Draft Runner

**Files:**
- Modify: `python/sglang/srt/model_executor/model_runner.py`
- Modify: `python/sglang/srt/speculative/eagle_worker.py`
- Test: `test/registered/unit/test_bench_one_batch_utils.py`

**Interfaces:**
- Adds: `ModelRunner(..., enable_hisparse_override: Optional[bool] = None)`.
- Adds: pure helper `draft_memory_pool_token_capacity(target_hot_capacity, target_logical_capacity, enable_hisparse)`.

- [ ] **Step 1: Write failing capacity tests**

Verify the helper returns target logical capacity when HiSparse is enabled and target hot capacity otherwise.

- [ ] **Step 2: Verify RED**

Run the helper suite and expect an attribute error for the missing helper.

- [ ] **Step 3: Implement runner isolation**

Set `ModelRunner.enable_hisparse` from the override when provided. In `EAGLEWorker`, copy the target `MemoryPoolConfig`, replace `max_total_num_tokens` with `target_allocator.size_full` for HiSparse, and pass `enable_hisparse_override=False` to the draft worker. Do not mutate shared `ServerArgs`.

- [ ] **Step 4: Run helper and import checks**

Run the helper suite and `python3 -m compileall -q python/sglang/srt/model_executor/model_runner.py python/sglang/srt/speculative/eagle_worker.py`.

Expected: both pass.

- [ ] **Step 5: Commit**

Commit: `git commit -am "spec: keep MTP draft KV dense with HiSparse target"`

### Task 3: HiSparse Candidate Allocation Transaction

**Files:**
- Modify: `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- Create: `test/registered/unit/mem_cache/test_hisparse_mtp_transaction.py`

**Interfaces:**
- Overrides: `HiSparseTokenToKVPoolAllocator.backup_state()`.
- Overrides: `HiSparseTokenToKVPoolAllocator.restore_state(state)`.
- Adds internal mapping-update tracking used by `alloc_extend()`.

- [ ] **Step 1: Write the GPU-independent transaction test**

Construct an allocator through `__new__` with fake nested allocators and a small fake mapping object. Verify backup, mapping mutation registration, nested allocation mutation, and restore return all nested availability and mapping entries to their original values while retaining unrelated persistent mappings.

- [ ] **Step 2: Verify RED in a PyTorch-enabled environment**

Run: `PYTHONPATH=python python3 test/registered/unit/mem_cache/test_hisparse_mtp_transaction.py`

Expected: failure because HiSparse backup/restore still uses uninitialized top-level `free_pages`.

- [ ] **Step 3: Implement transaction state**

Back up both nested allocator states. While a transaction is active, record each logical index and its prior mapping before `alloc_extend()` writes the mapping. Restore mapping updates in reverse order, restore both allocators, and reject nested transactions.

- [ ] **Step 4: Run transaction and helper tests**

Run both focused test files. Expected: all pass.

- [ ] **Step 5: Commit**

Commit: `git commit -am "mem: make HiSparse candidate allocation transactional"`

### Task 4: Integrated Regression And Smoke Readiness

**Files:**
- Modify: `python/sglang/bench_one_batch.py`
- Modify: `docs/superpowers/specs/2026-07-23-one-batch-mtp-hisparse-design.md` only if implementation reveals a corrected invariant.

**Interfaces:**
- Existing one-batch CLI accepts `--enable-hisparse` with EAGLE MTP.
- Existing HBM report keeps target and draft KV rows separate.

- [ ] **Step 1: Add runtime invariants**

Before decode, assert target has a HiSparse coordinator, draft has no coordinator, and draft KV pool capacity covers the highest logical target location used by admitted requests.

- [ ] **Step 2: Run local verification**

Run:

```bash
PYTHONPATH=python python3 test/registered/unit/test_bench_one_batch_utils.py
python3 -m compileall -q \
  python/sglang/bench_one_batch.py \
  python/sglang/bench_one_batch_utils.py \
  python/sglang/srt/mem_cache/hisparse_memory_pool.py \
  python/sglang/srt/model_executor/model_runner.py \
  python/sglang/srt/speculative/eagle_worker.py
git diff --check
```

Expected: tests pass, compilation succeeds, and `git diff --check` is empty.

- [ ] **Step 3: Review the complete diff**

Confirm no online scheduling code, DeepEP dispatcher, output clipping, or metric formulas changed.

- [ ] **Step 4: Commit and push**

Commit: `git commit -am "feat: support one-batch MTP with HiSparse target KV"`

Push: `git push -u origin codex/one-batch-mtp-hisparse`
