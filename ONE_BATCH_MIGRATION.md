# One-batch migration to SGLang 0.5.21

The `one-batch-test-v0.5.21` branch starts at upstream
`e00930c5489053f26d86b179cee0d087f846acbb` (`v0.5.21`). The old implementation is
preserved at `af2988a6986e6fdb6082193b10346882c42486e3` on
`one-batch-test-v0.5.10`; the remote `one-batch-test` remains the old entry point.

## Environment and entry points

Use an independent CUDA 13 environment matching the v0.5.21 image (Torch 2.13.0,
FlashInfer 0.6.18 and sglang-kernel 0.4.7). Do not reuse the old Torch 2.9.1/cu129
DeepEP extension. Editable installation also builds Rust extensions: provide
Cargo/Rust and the required native build tools when using a runtime-only image.
Verify the source import path and Git commit on every node.

The implementation is `python -m sglang.benchmark.one_batch`. The legacy
`python -m sglang.bench_one_batch` entry point delegates to the same CLI.
The Paper-EtherNode launcher discovers the source layout without importing the
runtime and supports both version branches.

The custom benchmark preserves per-attention-DP target batch size, capacity-based
wave admission, chunked prefill, deferred HiSparse hydration, rank-consistent
warmup, exact profiler windows, core-forward TPOT and separate HBM accounting.
JSONL and profiler traces are written by global TP rank 0; save logs on every node.
MTP/speculative execution is deliberately unsupported by this benchmark.

## CPU checks

These focused tests do not require CUDA or a running model:

```bash
python3 -B test/registered/unit/test_bench_one_batch_utils.py
python3 -B test/registered/unit/test_one_batch_runtime_contracts.py
python3 -B test/registered/unit/layers/test_deepep_compat.py
python3 -B test/registered/unit/layers/test_deepep_v1_payload.py
python3 -B test/registered/unit/managers/test_hisparse_host_ready.py
```

The runtime tests execute extracted production methods with CPU protocol doubles.
They verify API and lifecycle contracts, not GPU DMA, graph capture or collective
correctness. The launcher and trace-analyzer suites live in Paper-EtherNode.

## GPU and multi-node acceptance: pending

The migration was developed on macOS without NVIDIA GPUs or a local Docker
runtime. Actual CUDA CLI parsing, model correctness, graph replay, communication
and performance must be validated in the target containers before treating this
branch as fully verified.

First compare a small model against unmodified v0.5.21, including ordinary KV,
chunking, graph/eager selection, repeated runs and cleanup. Then run these GLM-5.1
workloads with separate output directories for each version/run:

| Workload | Required observations |
| --- | --- |
| I4096/O128, 4 nodes, DP32, target BS/DP32 | Requested global BS1024, actual admitted BS separately recorded, 127 decode steps on full completion. |
| Decode profile | Capture steps `[16,48)` and exit after capture; preserve `decode/step_N/core_forward` ranges. |
| Prefill profile, 2 nodes, DP16 | Capture waves `[8,24)`; exit after 24 admitted waves, BS/DP24, global BS384, no decode. |
| HiSparse I32768/O128, BS/DP32 | ratio8, max tokens133120, explicit decode graph maximum40, capacity stop reason and restored pools. |
| HiSparse I131072/O128, BS/DP13 decode profile | ratio14, max tokens131136, requested global BS416, actual BS in JSONL/trace, 32 captured steps. |

Record all-node commits, model/config, driver/GPU topology, dependency versions,
HCA/GID settings, generated commands and actual communication backends.

## Model and backend boundaries

- DeepSeek-V4.1 + HiSparse remains rejected by upstream validation. Use ordinary KV.
- GLM-5.3 needs validation against its actual architecture/quantization/backend;
  a hybrid GLM5Next configuration is not covered by old GLM-5.1 DSA results.
- DeepEP v1 Config/payload compatibility and micro warmup apply only to `deepep`.
  `deepep_v2`, `megamoe` and `flashinfer_megamoe` retain their own upstream rules.
- Unknown communication or composite/virtual-only KV memory accounting is
  reported as unknown, including dependent categories, rather than as zero cost.
  Independent total/free/active measurements remain available.
- Early-exit profiler runs are trace captures, not full-generation throughput runs.
