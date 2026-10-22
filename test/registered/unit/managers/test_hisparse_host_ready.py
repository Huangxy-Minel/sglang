"""CPU ownership tests using real coordinator and allocator method bodies.

The module imports GPU-only kernels, so extract its classes rather than replacing
production behavior. NumPy supplies the small CPU tensor fixtures.
"""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

# Register CPU CI without importing sglang's GPU/model dependencies.
_ci_spec = importlib.util.spec_from_file_location(
    "ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
_ci_spec.loader.exec_module(_ci)
register_cpu_ci = _ci.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")
COORDINATOR = ROOT / "python/sglang/srt/managers/hisparse_coordinator.py"
ALLOCATOR = ROOT / "python/sglang/srt/mem_cache/allocator/hisparse.py"


class Tensor(np.ndarray):
    def to(self, *args, **kwargs):
        return self

    def numel(self):
        return self.size

    def copy_(self, other):
        self[:] = other


def tensor(data, **kwargs):
    return np.asarray(data).view(Tensor)


def extract_class(path, name, namespace, methods=None):
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    node.bases = []
    if methods is not None:
        node.body = [
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods
        ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class HotPool:
    """Finite allocator that detects duplicate frees and reuses released slots."""

    def __init__(self):
        self.live = set()
        self.freed = []

    def alloc(self, size):
        result = [n for n in range(1, 17) if n not in self.live][:size]
        assert len(result) == size, "hot pool exhausted"
        self.live.update(result)
        return tensor(result)

    def free(self, indices):
        for n in indices:
            assert int(n) in self.live, "hot slot double free"
            self.live.remove(int(n))
            self.freed.append(int(n))


class HostPool:
    def __init__(self):
        self.live = set(range(10, 18))
        self.loads = []

    def allocated_host_indices(self, mapping, index, allocated_len):
        return mapping[index, : int(allocated_len)]

    def free(self, indices):
        for n in indices:
            assert int(n) in self.live, "host slot double free"
            self.live.remove(int(n))

    def load_to_device_per_layer(self, pool, host, device, layer, **kwargs):
        self.loads.append((host.tolist(), device.tolist(), layer))


class TestHostReady(unittest.TestCase):
    def setUp(self):
        self.reductions = []

        def all_reduce(value, op, group):
            self.reductions.append((int(value.item()), op, group))
            value[...] = min(int(value.item()), self.remote_ready)

        torch = SimpleNamespace(
            tensor=tensor,
            int=int,
            int32=np.int32,
            unique=lambda a: tensor(np.unique(a)),
            cat=lambda arrays: tensor(np.concatenate(arrays)),
            arange=lambda *a, **kw: tensor(np.arange(*a)),
            distributed=SimpleNamespace(
                all_reduce=all_reduce, ReduceOp=SimpleNamespace(MIN="min")
            ),
        )
        self.backup_waits = []
        namespace = dict(
            torch=torch,
            logger=SimpleNamespace(debug=lambda *a: None),
            device_module=SimpleNamespace(current_stream=lambda: "current"),
        )
        cls = extract_class(COORDINATOR, "HiSparseCoordinator", namespace)
        self.coordinator = c = cls.__new__(cls)
        c.tp_group, c.tp_world_size = "tp", 1
        self.remote_ready = 2
        self.ready = [True, True]
        self.reqs = [
            SimpleNamespace(
                rid=f"r{i}",
                kv=SimpleNamespace(req_pool_idx=i, kv_allocated_len=4),
                extend_range=SimpleNamespace(end=4),
                hisparse_staging=True,
            )
            for i in range(2)
        ]
        c.ack_staging_queue = [
            (None, SimpleNamespace(query=lambda i=i: self.ready[i]), req)
            for i, req in enumerate(self.reqs)
        ]
        c.req_to_token_pool = SimpleNamespace(
            req_to_token=tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        )
        self.hot = HotPool()
        mapping = tensor([0] * 17)
        mapping[1:9] = self.hot.alloc(8)
        self.device = SimpleNamespace(
            page_size=2,
            layer_num=1,
            full_to_hisparse_device_index_mapping=mapping,
            _translate_loc_to_hisparse_device=lambda locs: mapping[locs],
            translate_loc_from_full_to_compressed=lambda locs: locs,
        )
        alloc_cls = extract_class(
            ALLOCATOR,
            "HiSparseTokenToKVPoolAllocator",
            namespace,
            {"free_hisparse", "free_hisparse_indices", "alloc_device_buffer"},
        )
        allocator = alloc_cls.__new__(alloc_cls)
        allocator._kvcache, allocator.hisparse_attn_allocator = self.device, self.hot
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator.page_size, allocator.device = 2, "cpu"
        c.token_to_kv_pool_allocator, c.mem_pool_device = allocator, self.device
        c.mem_pool_host = self.host = HostPool()
        c.req_to_host_pool = tensor(
            [[10, 11, 12, 13, -1, -1], [14, 15, 16, 17, -1, -1]]
        )
        c.req_to_host_pool_allocated_len = tensor([4, 4])
        c.req_to_device_buffer = tensor([[0] * 4] * 2)
        c.req_device_buffer_size = tensor([0, 0])
        c.req_device_buffer_tokens = tensor([[[-1] * 4] * 2])
        c.req_device_buffer_token_locs = tensor([[[-1] * 4] * 2])
        c._device_buffer_arange_i32 = tensor([0, 1])
        c._lru_init = tensor([0, 1])
        c.lru_slots = tensor([[[0, 1], [0, 1]]])
        c.device_buffer_size, c.padded_buffer_size = 2, 4
        c.is_dsv4_hisparse, c.is_m3_hisparse = False, False
        c._skip_first_backup = [False, False]
        c.decode_producer_stream = None
        c._has_pending_backup = True
        c._backup_done_event = SimpleNamespace(
            wait=lambda stream: self.backup_waits.append(stream)
        )

    def collect(self):
        self.assertTrue(
            hasattr(self.coordinator, "collect_host_ready_reqs"),
            "host-ready collection must release hot slots without hydration",
        )
        return self.coordinator.collect_host_ready_reqs()

    def test_host_ready_retains_paged_host_and_logical_slots_but_releases_hot(self):
        c = self.coordinator
        result = self.collect()
        self.assertEqual(result, self.reqs)
        self.assertEqual(self.hot.live, set())
        self.assertEqual(self.host.live, set(range(10, 18)))
        self.assertEqual(c.req_to_host_pool_allocated_len.tolist(), [4, 4])
        self.assertEqual(
            c.req_to_token_pool.req_to_token.tolist(), [[1, 2, 3, 4], [5, 6, 7, 8]]
        )
        self.assertEqual(c.req_device_buffer_size.tolist(), [0, 0])
        self.assertEqual(
            c.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping.tolist(),
            [0] * 17,
        )
        self.assertFalse(any(req.hisparse_staging for req in self.reqs))
        self.assertEqual(self.collect(), [])
        self.assertEqual(len(self.hot.freed), 8)

    def test_tp_min_pops_only_common_completed_prefix(self):
        c = self.coordinator
        c.tp_world_size, self.remote_ready = 2, 1
        self.assertEqual(self.collect(), [self.reqs[0]])
        self.assertEqual(self.reductions, [(2, "min", "tp")])
        self.assertEqual(len(c.ack_staging_queue), 1)
        self.assertTrue(self.reqs[1].hisparse_staging)
        self.assertEqual(self.hot.live, {5, 6, 7, 8})
        self.assertEqual(self.collect(), [self.reqs[1]])

    def test_unfinished_first_event_blocks_later_completed_events(self):
        self.ready = [False, True]
        self.assertEqual(self.collect(), [])
        self.assertEqual(self.hot.live, set(range(1, 9)))
        self.assertEqual(len(self.coordinator.ack_staging_queue), 2)

    def test_delayed_direct_admission_and_finish_release_every_resource_once(self):
        c = self.coordinator
        result = self.collect()
        for req in result:
            c.admit_request_direct(req)
        self.assertEqual(c.req_device_buffer_size.tolist(), [4, 4])
        self.assertTrue(all(c._skip_first_backup))
        self.assertEqual(c.req_device_buffer_tokens.tolist(), [[[-1] * 4, [-1] * 4]])
        self.assertEqual(len(self.hot.live), 8)
        for req in result:
            c.request_finished(req)
        self.assertEqual(self.hot.live, set())
        self.assertEqual(self.host.live, set())
        self.assertEqual(c.req_to_host_pool_allocated_len.tolist(), [0, 0])
        self.assertEqual(c.req_device_buffer_size.tolist(), [0, 0])
        self.assertEqual(c.req_to_device_buffer.tolist(), [[0] * 4, [0] * 4])
        self.assertEqual(self.backup_waits, ["current"])
        self.assertFalse(c._has_pending_backup)

    def test_host_ready_cleanup_before_hydration_does_not_double_free_hot(self):
        c = self.coordinator
        for req in self.collect():
            c.request_finished(req)
        self.assertEqual(self.hot.live, set())
        self.assertEqual(len(self.hot.freed), 8)
        self.assertEqual(self.host.live, set())

    def test_paged_host_tail_is_retained_and_freed_by_recorded_allocation_length(self):
        c = self.coordinator
        c.req_to_host_pool[0, 4:] = [18, 19]
        c.req_to_host_pool_allocated_len[0] = 6
        self.host.live.update([18, 19])
        result = self.collect()
        self.assertEqual(c.req_to_host_pool[0].tolist(), [10, 11, 12, 13, 18, 19])
        self.assertTrue({18, 19}.issubset(self.host.live))
        for req in result:
            c.request_finished(req)
        self.assertEqual(self.host.live, set())

    def test_destroy_drains_prefetch_and_backup_before_host_pool_destroy(self):
        c = self.coordinator
        events = []
        c.write_staging_stream = SimpleNamespace(
            synchronize=lambda: events.append("staging")
        )
        c.decode_backup_stream = SimpleNamespace(
            synchronize=lambda: events.append("backup")
        )
        c.enable_prefetch = True
        c.prefetch_stream = SimpleNamespace(
            synchronize=lambda: events.append("prefetch")
        )
        self.host.destroy = lambda: events.append("host-destroy")
        c.destroy()
        self.assertEqual(events, ["staging", "backup", "prefetch", "host-destroy"])

    def test_ordinary_serving_collector_still_reuses_prefill_as_decode_buffer(self):
        c = self.coordinator
        result = c.collect_ready_reqs()
        self.assertEqual(result, self.reqs)
        self.assertEqual(c.req_device_buffer_size.tolist(), [4, 4])
        self.assertEqual(
            c.req_device_buffer_tokens[0, :, :2].tolist(), [[0, 1], [0, 1]]
        )
        self.assertEqual(len(self.hot.live), 8)
        self.assertEqual(self.host.live, set(range(10, 18)))
        self.assertFalse(any(req.hisparse_staging for req in result))
        self.assertTrue(all(c._skip_first_backup))


if __name__ == "__main__":
    unittest.main()
