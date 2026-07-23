# One-Batch MTP + HiSparse Design

## Scope

Enable `MTP + HiSparse` in CUDA one-batch benchmarks while preserving the
existing online allocator and scheduler behavior. The target model uses
HiSparse. The MTP draft model keeps a dense device KV cache.

The implementation continues to support built-in EAGLE V1 MTP checkpoints.
It does not add draft KV host staging, external draft models, correctness mode,
or eager-mode profiling.

## Memory Ownership

- Target KV uses the existing HiSparse logical, host, and hot-device pools.
- Draft KV uses a separate dense device pool indexed by the target logical
  token locations. Its capacity therefore matches the target logical pool,
  not the smaller target hot pool.
- Target and draft runners continue to share the request-to-token pool and the
  target token allocator. They do not share KV tensors.
- Speculative candidate locations are temporary. A cycle allocates candidate
  locations, writes candidate draft KV, verifies them, commits the accepted
  path, and returns rejected or unused locations to the allocator.
- The next cycle may reuse the same physical locations, but address identity is
  not part of the interface.

## Runner Initialization

The target runner initializes HiSparse normally, including its coordinator.
The EAGLE draft runner is explicitly marked as target-only-HiSparse mode:

- It must not construct a `HiSparseNSATokenToKVPool`.
- It must not construct a `HiSparseCoordinator`.
- It constructs a dense draft KV pool with target logical capacity.
- It retains the shared target allocator so request token locations stay
  identical between target and draft KV pools.

This mode is internal to `EAGLEWorker`; it does not change CLI configuration or
online non-speculative execution.

## Allocator Transactions

EAGLE draft candidate generation uses allocator backup and restore. The
HiSparse allocator must make that transaction cover all mutable state:

- logical allocator free state;
- hot allocator free state;
- logical-to-hot mapping entries changed by the temporary allocation.

Restoring a transaction must release temporary logical/hot locations and
restore only the touched mapping entries. Persistent request mappings from
earlier cycles must remain unchanged.

Normal `alloc_extend`, staging, hydration, and free behavior remain unchanged.

## One-Batch Flow

Wave prefill remains unchanged except that MTP and HiSparse may now be enabled
together:

1. Target prefill writes target KV into the HiSparse path.
2. Draft extend writes dense draft KV at the same logical token locations.
3. The completed target request is staged to host.
4. Ready requests are hydrated into the target hot buffer before decode.
5. Each decode cycle runs EAGLE draft, target verify, accepted-path commit, and
   draft extend.

Only target KV participates in HiSparse staging and hydration. Draft KV remains
resident in HBM for the admitted context.

## Admission

HiSparse wave admission must reserve all of the following per ready request:

- target logical capacity for `input_len + output_len`;
- target host capacity for the same lifetime;
- target hot decode buffer and prefill peak;
- target speculative candidate reserve;
- dense draft KV capacity for the committed context plus speculative reserve;
- one request-pool slot.

The benchmark stops before launching a wave when any rank cannot satisfy the
next request. New stop reasons distinguish target capacity from draft dense KV
capacity.

## Observability

The existing HBM report continues to list target KV and draft KV separately.
Wave admission diagnostics include draft required/available token capacity when
MTP is active. Decode metrics and exact output-length accounting are unchanged.

## Validation

Unit tests cover:

- accepting `MTP + HiSparse` while retaining existing unsupported-combination
  checks;
- target-only HiSparse draft-runner configuration;
- HiSparse allocator transaction restore without damaging persistent mappings;
- admission rejection when dense draft KV capacity is insufficient;
- existing MTP and HiSparse helper regressions.

GPU smoke tests cover GLM-5.1-FP8 with one-batch MTP + HiSparse, first at a
small batch and then at the intended multi-node configuration.
