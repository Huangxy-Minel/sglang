# Blackwell DeepEP Payload-Byte Compatibility Design

## Context

The custom Blackwell DeepEP build selects its normal dispatch configuration
using `real_hidden_bytes`. SGLang currently uses one BF16-oriented value,
`hidden_size * 2`, for both normal dispatch and combine buffer sizing.

With GLM-5.1-FP8 and DeepGEMM enabled, SGLang quantizes normal-dispatch
activations to FP8 before calling DeepEP. The real dispatch payload is therefore
`hidden_size * 1` bytes, while combine output remains BF16 at
`hidden_size * 2` bytes.

Passing the shared BF16 size to Blackwell DeepEP causes:

```text
get_dispatch_config real_hidden_bytes=12288
actual real_hidden_bytes=6144
```

Forcing BF16 dispatch makes those values agree, but removes the FP8 scale tensor
required by the current DeepGEMM masked-GEMM path. BF16 dispatch is therefore a
diagnostic confirmation, not a viable fix for this configuration.

## Approaches Considered

### 1. Separate dispatch and combine payload sizes using one shared policy

Determine whether normal dispatch uses FP8 with the same policy used by
`dispatch_a`. Use one byte per hidden element for FP8 dispatch and two bytes per
hidden element for BF16 dispatch. Continue using two bytes per hidden element
for BF16 combine.

This is the selected approach. It is small, supports both FP8 and BF16 dispatch,
and keeps dispatch behavior and buffer sizing consistent through one helper.

### 2. Derive dispatch bytes from the runtime tensor

Inspect the tensor passed to `_dispatch_core` and calculate its element size.
This is exact for the current call, but it complicates the singleton buffer's
initialization in `auto` mode, where normal and low-latency capacity can be
allocated together before a normal-dispatch tensor is available.

### 3. Hard-code Blackwell dispatch to one byte

Always pass `hidden_size` to `get_dispatch_config`. This fixes the observed FP8
case but breaks BF16 dispatch and Cutlass/non-DeepGEMM paths. It is rejected.

## Selected Design

Add pure compatibility helpers that:

1. Decide whether normal DeepEP dispatch uses FP8 from:
   - DeepGEMM JIT being enabled;
   - the MoE runner not being Cutlass;
   - `SGLANG_DEEPEP_BF16_DISPATCH` being disabled.
2. Convert `hidden_size` and the dispatch mode into the real dispatch payload
   byte count.

Both `dispatch_a` and DeepEP buffer initialization use this shared decision.
This prevents the quantization path and the Blackwell config path from
diverging.

During normal buffer initialization:

- obtain the dispatch config with the dispatch payload bytes;
- size dispatch buffer hints with the dispatch payload bytes;
- size combine buffer hints with BF16 combine payload bytes;
- retain the maximum NVLink and RDMA hints across both configs.

Low-latency sizing remains unchanged.

## Compatibility

The existing wrapper continues to inspect the installed DeepEP signature:

- custom Blackwell DeepEP receives `real_hidden_bytes`;
- legacy DeepEP continues to receive only `num_ranks`.

No DeepEP or NVSHMEM source changes are required.

## Tests

Unit tests cover:

1. FP8 normal dispatch for hidden size 6144 produces 6144 bytes.
2. BF16 normal dispatch for hidden size 6144 produces 12288 bytes.
3. Cutlass/non-DeepGEMM paths remain BF16.
4. Blackwell DeepEP receives the calculated dispatch bytes.
5. Legacy DeepEP remains compatible.
6. Dispatch and combine buffer hints are evaluated with their respective byte
   counts.

The server-side acceptance test is the existing GLM-5.1-FP8 one-batch run with
`SGLANG_DEEPEP_BF16_DISPATCH` unset. It must pass DeepEP normal dispatch without
either the payload-size assertion or a missing FP8 scale.
