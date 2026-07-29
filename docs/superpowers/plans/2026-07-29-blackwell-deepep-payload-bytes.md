# Blackwell DeepEP Payload-Byte Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SGLang pass the real FP8 normal-dispatch payload size to custom Blackwell DeepEP while preserving BF16 combine buffer capacity and legacy DeepEP compatibility.

**Architecture:** Keep DeepEP signature compatibility and payload-size policy in the dependency-free `deepep_compat.py` module. Make `deepep.py` use one shared FP8-dispatch decision for both activation quantization and buffer sizing, while calculating dispatch and combine buffer hints independently.

**Tech Stack:** Python 3.12+, `unittest`, SGLang DeepEP token dispatcher.

## Global Constraints

- FP8 normal dispatch for hidden size 6144 uses 6144 payload bytes.
- BF16 normal dispatch and combine for hidden size 6144 use 12288 payload bytes.
- Legacy DeepEP continues receiving only `num_ranks`.
- Low-latency DeepEP sizing remains unchanged.
- Do not modify DeepEP or NVSHMEM source.

---

### Task 1: Add payload-size policy and independent buffer-hint helpers

**Files:**
- Modify: `python/sglang/srt/layers/moe/token_dispatcher/deepep_compat.py`
- Test: `test/registered/unit/layers/test_deepep_compat.py`

**Interfaces:**
- Produces: `use_fp8_normal_dispatch(enable_jit_deepgemm: bool, is_cutlass: bool, force_bf16_dispatch: bool) -> bool`
- Produces: `get_normal_dispatch_hidden_bytes(hidden_size: int, use_fp8_dispatch: bool) -> int`
- Produces: `get_normal_buffer_size_hints(dispatch_config, combine_config, dispatch_hidden_bytes: int, combine_hidden_bytes: int, num_ranks: int) -> tuple[int, int]`

- [x] **Step 1: Write failing policy and sizing tests**

Add tests asserting:

```python
self.assertTrue(
    compat.use_fp8_normal_dispatch(
        enable_jit_deepgemm=True,
        is_cutlass=False,
        force_bf16_dispatch=False,
    )
)
self.assertFalse(
    compat.use_fp8_normal_dispatch(
        enable_jit_deepgemm=True,
        is_cutlass=False,
        force_bf16_dispatch=True,
    )
)
self.assertFalse(
    compat.use_fp8_normal_dispatch(
        enable_jit_deepgemm=True,
        is_cutlass=True,
        force_bf16_dispatch=False,
    )
)
self.assertEqual(
    compat.get_normal_dispatch_hidden_bytes(6144, use_fp8_dispatch=True),
    6144,
)
self.assertEqual(
    compat.get_normal_dispatch_hidden_bytes(6144, use_fp8_dispatch=False),
    12288,
)
```

Use deterministic fake configs whose hint return values incorporate the supplied
payload bytes, and assert that `get_normal_buffer_size_hints` uses 6144 for
dispatch and 12288 for combine.

- [x] **Step 2: Run tests and verify the new behavior fails**

Run:

```bash
python3 -m unittest discover \
  -s test/registered/unit/layers \
  -p 'test_deepep_compat.py' \
  -v
```

Expected: existing legacy/Blackwell signature tests pass; new tests fail because
the three new helpers do not exist.

- [x] **Step 3: Implement the minimal pure helpers**

Add:

```python
def use_fp8_normal_dispatch(
    enable_jit_deepgemm: bool,
    is_cutlass: bool,
    force_bf16_dispatch: bool,
) -> bool:
    return enable_jit_deepgemm and not is_cutlass and not force_bf16_dispatch


def get_normal_dispatch_hidden_bytes(
    hidden_size: int,
    use_fp8_dispatch: bool,
) -> int:
    return hidden_size * (1 if use_fp8_dispatch else 2)
```

Implement `get_normal_buffer_size_hints` by evaluating dispatch hints with
`dispatch_hidden_bytes`, combine hints with `combine_hidden_bytes`, and
returning the maximum NVLink and RDMA values.

- [x] **Step 4: Run tests and verify all compatibility tests pass**

Run the command from Step 2.

Expected: all tests pass.

---

### Task 2: Integrate the shared payload policy into DeepEP normal mode

**Files:**
- Modify: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:1-190`
- Modify: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:320-440`
- Test: `test/registered/unit/layers/test_deepep_compat.py`

**Interfaces:**
- Consumes: the three helpers from Task 1.
- Produces: `_use_fp8_normal_dispatch() -> bool`, used by both quantization and buffer sizing.
- Changes: `DeepEPBuffer.get_deepep_buffer` accepts `use_fp8_dispatch: bool`.

- [x] **Step 1: Add an integration-oriented failing sizing test**

Extend the compatibility test so a Blackwell config selected with
`dispatch_hidden_bytes=6144` and a combine config sized with
`combine_hidden_bytes=12288` returns the literal expected maximum hints. Mutating
either input to 12288 or 6144 respectively must change the result and fail the
test.

- [x] **Step 2: Run the focused test and verify failure**

Run:

```bash
python3 test/registered/unit/layers/test_deepep_compat.py
```

Expected: failure until the independent hint calculation is present.

- [x] **Step 3: Integrate shared policy and independent payload sizes**

In `deepep.py`:

1. Import the three Task 1 helpers.
2. Add `_use_fp8_normal_dispatch()` that passes the live DeepGEMM, Cutlass, and
   BF16-dispatch settings to `use_fp8_normal_dispatch`.
3. Replace the inline condition in `dispatch_a` with
   `_use_fp8_normal_dispatch()`.
4. Pass the `_use_fp8_normal_dispatch()` result into buffer initialization.
5. In `DeepEPBuffer.get_deepep_buffer`, calculate:

```python
dispatch_hidden_bytes = get_normal_dispatch_hidden_bytes(
    hidden_size,
    use_fp8_dispatch,
)
combine_hidden_bytes = hidden_size * param_bytes
```

6. Select the Blackwell dispatch config with `dispatch_hidden_bytes`.
7. Evaluate dispatch/combine buffer hints independently through
   `get_normal_buffer_size_hints`.
8. Leave low-latency size-hint logic unchanged.

- [x] **Step 4: Run focused tests and syntax checks**

Run:

```bash
python3 -m unittest discover \
  -s test/registered/unit/layers \
  -p 'test_deepep_compat.py' \
  -v

PYTHONPYCACHEPREFIX=/tmp/sglang-pyc python3 -m compileall -q \
  python/sglang/srt/layers/moe/token_dispatcher/deepep.py \
  python/sglang/srt/layers/moe/token_dispatcher/deepep_compat.py \
  test/registered/unit/layers/test_deepep_compat.py

git diff --check
```

Expected: all commands exit zero.

- [x] **Step 5: Commit the implementation**

```bash
git add \
  python/sglang/srt/layers/moe/token_dispatcher/deepep.py \
  python/sglang/srt/layers/moe/token_dispatcher/deepep_compat.py \
  test/registered/unit/layers/test_deepep_compat.py \
  docs/superpowers/plans/2026-07-29-blackwell-deepep-payload-bytes.md

git commit -m "fix: size Blackwell DeepEP FP8 dispatch correctly"
```

The B200 server acceptance test remains external: pull the commit, unset
`SGLANG_DEEPEP_BF16_DISPATCH`, restart all workers, and rerun the GLM-5.1-FP8
one-batch command.
