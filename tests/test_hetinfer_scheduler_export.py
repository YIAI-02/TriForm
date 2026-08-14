from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cost_model import CostModel  # noqa: E402
from hardware import Cluster, DeviceSpec, LinkSpec  # noqa: E402
import scheduler.scheduler_bifocal as bifocal_module  # noqa: E402
from scheduler.scheduler_bifocal import BifocalScheduler  # noqa: E402
from task_graph import TaskGraph, TaskNode  # noqa: E402


DECISION_CANARY_S = 1000.0


class _FixtureCost:
    """Deterministic primitive costs while production capture stays intact."""

    def __init__(self, cluster: Cluster) -> None:
        self.cluster = cluster

    def get_host_device(self) -> DeviceSpec:
        return self.cluster.devices["CPU0"]

    @staticmethod
    def estimate_activation_bytes(
        node: TaskNode, batch: int, seq_len: int, phase: str
    ) -> tuple[int, int]:
        del batch, seq_len, phase
        if bool(node.attrs.get("fixture_export_failure", False)):
            raise RuntimeError("fixture export finalization failure")
        return (int(node.bytes_read), int(node.bytes_write))

    @staticmethod
    def device_preferred_fmt(device: DeviceSpec) -> str:
        if str(device.type).lower() == "pim":
            return "PIM_BLOCKED"
        return "ND"

    @staticmethod
    def format_size(bytes_nd: int, layout: str) -> int:
        del layout
        return int(bytes_nd)

    @staticmethod
    def format_conversion_time(
        bytes_amount: int,
        source_layout: str,
        destination_layout: str,
        device: DeviceSpec,
    ) -> float:
        del bytes_amount, source_layout, destination_layout, device
        return 0.0

    @staticmethod
    def activation_read_time_pim(bytes_amount: int) -> float:
        del bytes_amount
        return 0.0

    @staticmethod
    def mem_time(bytes_amount: int, device: DeviceSpec) -> float:
        return float(bytes_amount) / (
            float(device.mem_bw_GBs) * 1024.0 * 1024.0 * 1024.0
        )

    @staticmethod
    def comm_cost(
        source: DeviceSpec, destination: DeviceSpec, bytes_amount: int
    ) -> float:
        if source.name == destination.name:
            return 0.0
        return float(bytes_amount) / 1_000_000_000.0

    def combine_transfer_and_convert(
        self,
        source: DeviceSpec,
        destination: DeviceSpec,
        bytes_amount: int,
        source_layout: str,
        destination_layout: str,
    ) -> float:
        del source_layout, destination_layout
        return self.comm_cost(source, destination, bytes_amount)


class _FixtureBifocal(BifocalScheduler):
    """Exercise real placement/capture control flow with explicit timing canaries."""

    def __init__(
        self,
        *,
        rand_seed: int,
        fail_on_commit: str | None = None,
        include_second_npu: bool = False,
    ) -> None:
        cluster = Cluster()
        cluster.topology = "fc"
        cluster.add_device(
            DeviceSpec(
                name="CPU0",
                type="cpu",
                tflops=1.0,
                mem_bw_GBs=100.0,
                mem_capacity_GB=1.0,
            )
        )
        cluster.add_device(
            DeviceSpec(
                name="NPU0",
                type="npu",
                tflops=10.0,
                mem_bw_GBs=1000.0,
                mem_capacity_GB=1.0,
            )
        )
        if include_second_npu:
            cluster.add_device(
                DeviceSpec(
                    name="NPU1",
                    type="npu",
                    tflops=10.0,
                    mem_bw_GBs=1000.0,
                    mem_capacity_GB=1.0,
                )
            )
        cluster.add_device(
            DeviceSpec(
                name="PIM0",
                type="pim",
                tflops=5.0,
                mem_bw_GBs=500.0,
                mem_capacity_GB=1.0,
            )
        )
        self._fixture_fail_on_commit = fail_on_commit
        super().__init__(
            cluster=cluster,
            cost=_FixtureCost(cluster),
            label=SimpleNamespace(
                kv_place="host",
                kv_in_pim=False,
                kv_total_bytes=0,
            ),
            batch=1,
            seq_len=8,
            buffer=None,
            rand_seed=rand_seed,
        )

    def reset_state(self, *, clear_caches: bool = True) -> None:
        super().reset_state(clear_caches=clear_caches)
        # Each fixture run is an independent placement trial. This lets the
        # same scheduler object compare capture off/on without phase carry-over.
        self.avail = {name: 0.0 for name in self.cluster.devices}

    def _upward_rank(self, g: TaskGraph, phase: str) -> list[str]:
        idx = self._get_graph_index(g)
        count = len(idx.nodes)
        idx.rank_u_by_phase[phase] = {
            nid: float(count - offset) for offset, nid in enumerate(idx.topo)
        }
        return list(idx.topo)

    def _weighted_compute_time(
        self,
        node: TaskNode,
        device: DeviceSpec,
        label: object,
        batch: int,
        seq_len: int,
        phase: str,
    ) -> float:
        del label, batch, seq_len, phase
        return float(node.attrs["fixture_compute_s"][device.name])

    def _earliest_finish_on_device(
        self,
        g: TaskGraph,
        nid: str,
        device: DeviceSpec,
        label: object,
        phase: str,
        commit: bool,
    ) -> tuple[float, float]:
        if self._is_comm_node(g.nodes[nid]):
            return super()._earliest_finish_on_device(
                g, nid, device, label, phase, commit
            )
        del label, phase
        if commit and nid == self._fixture_fail_on_commit:
            raise RuntimeError(f"fixture commit failure for {nid}")
        predecessor_finish = max(
            (
                float(self._node_finish_time.get(pred, 0.0))
                for pred in g.predecessors(nid)
            ),
            default=0.0,
        )
        start = max(predecessor_finish, float(self.avail.get(device.name, 0.0)))
        # The canary stands in for queue, reload, and communication delay in the
        # placement/EFT path. It must never leak into exported local service.
        duration = (
            float(g.nodes[nid].attrs["fixture_compute_s"][device.name])
            + DECISION_CANARY_S
        )
        finish = start + duration
        if commit:
            self._node_finish_time[nid] = finish
            self._node_placement[nid] = str(device.name)
            self._node_out_fmt[nid] = str(self.cost.device_preferred_fmt(device))
            self.avail[device.name] = finish
        return (start, finish)

    def _after_commit_consume_predecessors(self, g: TaskGraph, nid: str) -> None:
        del g, nid


def _node(
    node_id: str,
    *,
    read_bytes: int,
    write_bytes: int,
    npu_compute_s: float,
    pim_compute_s: float | None,
    external_input: bool = False,
) -> TaskNode:
    allowed = {"cpu": False, "npu": True, "pim": pim_compute_s is not None}
    compute = {"NPU0": float(npu_compute_s)}
    if pim_compute_s is not None:
        compute["PIM0"] = float(pim_compute_s)
    attrs: dict[str, object] = {"fixture_compute_s": compute}
    if external_input:
        attrs["hetinfer_external_inputs"] = [
            {
                "tensor_id": "request_input",
                "source_devices": ["NPU0"],
                "bytes": 4096,
                "layout": "ND",
            }
        ]
    return TaskNode(
        id=node_id,
        name=node_id.upper(),
        bytes_read=read_bytes,
        bytes_write=write_bytes,
        allowed=allowed,
        attrs=attrs,
    )


def _graph() -> TaskGraph:
    """Fork/join with deliberately mismatched producer-write/consumer-read sizes."""

    graph = TaskGraph()
    graph.add_node(
        _node(
            "source",
            read_bytes=111,
            write_bytes=32_768,
            npu_compute_s=0.001,
            pim_compute_s=None,
            external_input=True,
        )
    )
    graph.add_node(
        _node(
            "left",
            read_bytes=12_345,
            write_bytes=65_536,
            npu_compute_s=0.002,
            pim_compute_s=0.002,
        )
    )
    graph.add_node(
        _node(
            "right",
            read_bytes=54_321,
            write_bytes=131_072,
            npu_compute_s=0.004,
            pim_compute_s=0.004,
        )
    )
    graph.add_node(
        _node(
            "join",
            read_bytes=777,
            write_bytes=8192,
            npu_compute_s=0.006,
            pim_compute_s=0.007,
        )
    )
    graph.add_edge("source", "left")
    graph.add_edge("source", "right")
    graph.add_edge("left", "join")
    graph.add_edge("right", "join")
    return graph


FIXED_PLAN = {
    "order": ("source", "left", "right", "join"),
    "device_by_node": {
        "source": "NPU0",
        "left": "PIM0",
        "right": "NPU0",
        "join": "NPU0",
    },
}


def _schedule_signature(schedule: list[object]) -> list[tuple[object, ...]]:
    return [
        (
            task.node_id,
            task.device,
            float(task.start),
            float(task.finish),
        )
        for task in schedule
    ]


def _run(
    scheduler: _FixtureBifocal,
    graph: TaskGraph,
    *,
    fixed: bool,
) -> list:
    if fixed:
        return scheduler.schedule_with_plan(graph, "prefill", FIXED_PLAN)
    with mock.patch.object(bifocal_module, "SCHED_JOINT_LK_ENABLE", False):
        return scheduler.schedule(graph, "prefill")


class HetInferSchedulerExportTests(unittest.TestCase):
    def _assert_clean_complete_snapshot(
        self,
        snapshot: dict,
        scheduler: _FixtureBifocal,
        schedule: list,
        graph: TaskGraph,
    ) -> None:
        self.assertEqual(
            set(snapshot),
            {
                "schedule_call_index",
                "phase",
                "devices",
                "operators",
                "inputs",
                "collective_contexts",
                "routes",
            },
        )
        self.assertEqual(snapshot["schedule_call_index"], 1)
        self.assertEqual(snapshot["phase"], "prefill")
        self.assertEqual(snapshot["collective_contexts"], [])
        device_type_by_id = {
            device["device_id"]: device["device_type"]
            for device in snapshot["devices"]
        }
        for device in snapshot["devices"]:
            self.assertEqual(set(device), {"device_id", "device_type"})

        operator_by_node = {
            entry["op_id"].rsplit(":", 1)[-1]: entry
            for entry in snapshot["operators"]
        }
        self.assertEqual(set(operator_by_node), set(graph.nodes))
        schedule_by_node = {task.node_id: task for task in schedule}
        for node_id, operator in operator_by_node.items():
            self.assertEqual(
                set(operator),
                {
                    "op_id",
                    "dependencies",
                    "legal_devices",
                    "expert_device",
                    "service_s",
                    "atlas_descriptor",
                },
            )
            self.assertEqual(set(operator["service_s"]), set(operator["legal_devices"]))
            self.assertEqual(
                operator["expert_device"], scheduler._node_placement[node_id]
            )
            self.assertEqual(
                operator["expert_device"], schedule_by_node[node_id].device
            )

            # Exported service is pure local compute. The placement duration
            # contains a 1000-second queue/reload/communication canary.
            for device_name, service_s in operator["service_s"].items():
                if device_name == "PIM0":
                    self.assertIsNone(service_s)
                else:
                    self.assertEqual(
                        service_s,
                        graph.nodes[node_id].attrs["fixture_compute_s"][device_name],
                    )
                    self.assertLess(service_s, DECISION_CANARY_S)
            task = schedule_by_node[node_id]
            chosen_compute = graph.nodes[node_id].attrs["fixture_compute_s"][
                task.device
            ]
            self.assertAlmostEqual(
                task.finish - task.start,
                chosen_compute + DECISION_CANARY_S,
                places=10,
            )
            descriptor = operator["atlas_descriptor"]
            self.assertEqual(
                set(descriptor),
                {
                    "op_kind",
                    "phase",
                    "batch",
                    "seq_len",
                    "attrs",
                    "weight_layout_by_device",
                    "collective_primitive",
                    "collective_participants",
                    "topology",
                },
            )
            self.assertEqual(descriptor["phase"], "prefill")
            self.assertEqual(descriptor["batch"], 1)
            self.assertEqual(descriptor["seq_len"], 8)
            self.assertEqual(descriptor["topology"], "fc")
            self.assertIsNone(descriptor["collective_primitive"])
            self.assertEqual(descriptor["collective_participants"], [])
            self.assertEqual(
                set(descriptor["weight_layout_by_device"]),
                set(operator["legal_devices"]),
            )
            self.assertEqual(
                set(descriptor["weight_layout_by_device"].values()), {"NONE"}
            )

        self.assertEqual(operator_by_node["source"]["dependencies"], [])
        self.assertEqual(operator_by_node["left"]["dependencies"], ["prefill:1:source"])
        self.assertEqual(
            operator_by_node["right"]["dependencies"], ["prefill:1:source"]
        )
        self.assertEqual(
            operator_by_node["join"]["dependencies"],
            ["prefill:1:left", "prefill:1:right"],
        )

        for route in snapshot["routes"]:
            self.assertEqual(
                set(route),
                {
                    "tensor_id",
                    "source_device_id",
                    "destination_device_id",
                    "bytes",
                    "layout",
                    "duration_s",
                    "requires_atlas",
                    "atlas_descriptor",
                },
            )
            self.assertEqual(
                route["atlas_descriptor"],
                {
                    "topology": "fc",
                    "source_device_type": device_type_by_id[
                        route["source_device_id"]
                    ],
                    "destination_device_type": device_type_by_id[
                        route["destination_device_id"]
                    ],
                },
            )
            resident = route["source_device_id"] == route["destination_device_id"]
            pim_related = "PIM0" in {
                route["source_device_id"],
                route["destination_device_id"],
            }
            if resident:
                self.assertEqual(route["duration_s"], 0.0)
                self.assertFalse(route["requires_atlas"])
            elif pim_related:
                self.assertIsNone(route["duration_s"])
                self.assertTrue(route["requires_atlas"])
            else:
                self.assertGreater(route["duration_s"], 0.0)
                self.assertFalse(route["requires_atlas"])

        route_keys = {
            (
                route["tensor_id"],
                route["source_device_id"],
                route["destination_device_id"],
                route["bytes"],
                route["layout"],
            )
            for route in snapshot["routes"]
        }
        self.assertEqual(
            {
                key
                for key in route_keys
                if key[0] == "prefill:1:request_input"
            },
            {("prefill:1:request_input", "NPU0", "NPU0", 4096, "ND")},
        )

        def expected_closure(
            producer: str,
            producer_sources: set[str],
            consumer_destinations: set[str],
            bytes_: int,
        ) -> set[tuple[str, str, str, int, str]]:
            tensor = f"prefill:1:tensor:{producer}"
            return {
                (
                    tensor,
                    source,
                    destination,
                    bytes_,
                    "PIM_BLOCKED" if source == "PIM0" else "ND",
                )
                for source in producer_sources | {"CPU0"}
                for destination in consumer_destinations
            }

        expected_routes = {
            ("prefill:1:request_input", "NPU0", "NPU0", 4096, "ND")
        }
        expected_routes |= expected_closure(
            "source", {"NPU0"}, {"CPU0", "NPU0", "PIM0"}, 32_768
        )
        expected_routes |= expected_closure(
            "left", {"NPU0", "PIM0"}, {"CPU0", "NPU0", "PIM0"}, 65_536
        )
        expected_routes |= expected_closure(
            "right", {"NPU0", "PIM0"}, {"CPU0", "NPU0", "PIM0"}, 131_072
        )
        self.assertEqual(route_keys, expected_routes)

        input_by_edge = {
            (
                entry["producer_op_id"],
                entry["consumer_op_id"],
                entry["tensor_id"],
            ): entry
            for entry in snapshot["inputs"]
        }
        self.assertEqual(len(input_by_edge), 5)
        for entry in snapshot["inputs"]:
            self.assertEqual(
                set(entry),
                {
                    "consumer_op_id",
                    "producer_op_id",
                    "tensor_id",
                    "semantics",
                    "bytes",
                    "source_residencies",
                    "destination_devices",
                },
            )
            self.assertEqual(entry["semantics"], "data")
            self.assertTrue(entry["source_residencies"])
            self.assertTrue(entry["destination_devices"])

        external = input_by_edge[
            (None, "prefill:1:source", "prefill:1:request_input")
        ]
        self.assertEqual(external["bytes"], 4096)
        self.assertEqual(
            external["source_residencies"],
            [{"device_id": "NPU0", "layout": "ND"}],
        )
        self.assertEqual(external["destination_devices"], ["NPU0"])

        expected_edge_inputs = {
            ("source", "left", 32_768),
            ("source", "right", 32_768),
            ("left", "join", 65_536),
            ("right", "join", 131_072),
        }
        for producer, consumer, expected_bytes in expected_edge_inputs:
            tensor_id = f"prefill:1:tensor:{producer}"
            entry = input_by_edge[
                (
                    f"prefill:1:{producer}",
                    f"prefill:1:{consumer}",
                    tensor_id,
                )
            ]
            self.assertEqual(entry["bytes"], expected_bytes)
            self.assertEqual(entry["destination_devices"], ["NPU0", "PIM0"])
            expected_sources = {"CPU0", "NPU0"}
            if producer in {"left", "right"}:
                expected_sources.add("PIM0")
            self.assertEqual(
                {
                    residency["device_id"]
                    for residency in entry["source_residencies"]
                },
                expected_sources,
            )

        # Host spill is a first-class intermediate residency: one route stores
        # the producer output and another reloads that same tensor to a later
        # accelerator candidate.
        self.assertIn(
            ("prefill:1:tensor:source", "NPU0", "CPU0", 32_768, "ND"),
            route_keys,
        )
        self.assertIn(
            ("prefill:1:tensor:source", "CPU0", "PIM0", 32_768, "ND"),
            route_keys,
        )

        # Both fork successors use the same default producer tensor identity;
        # their differing read sizes cannot alter the producer's 32768 bytes.
        source_tensor_routes = {
            key for key in route_keys if key[0] == "prefill:1:tensor:source"
        }
        self.assertEqual(len(source_tensor_routes), 6)
        self.assertEqual({key[3] for key in source_tensor_routes}, {32_768})
        self.assertNotEqual(graph.nodes["left"].bytes_read, 32_768)
        self.assertNotEqual(graph.nodes["right"].bytes_read, 32_768)
        self.assertEqual(
            scheduler._hetinfer_edge_tensor_id(graph, "source", "left", "prefill", 1),
            scheduler._hetinfer_edge_tensor_id(graph, "source", "right", "prefill", 1),
        )
        self.assertEqual(
            {
                key[3]
                for key in route_keys
                if key[0] == "prefill:1:tensor:left"
            },
            {65_536},
        )
        self.assertEqual(
            {
                key[3]
                for key in route_keys
                if key[0] == "prefill:1:tensor:right"
            },
            {131_072},
        )
        self.assertNotEqual(graph.nodes["join"].bytes_read, 65_536)
        self.assertNotEqual(graph.nodes["join"].bytes_read, 131_072)

    def test_schedule_and_fixed_plan_publish_clean_complete_snapshots(self) -> None:
        for fixed in (False, True):
            with self.subTest(path="fixed" if fixed else "schedule"):
                graph = _graph()
                scheduler = _FixtureBifocal(rand_seed=7)
                scheduler.enable_hetinfer_prior_capture(True)
                schedule = _run(scheduler, graph, fixed=fixed)
                snapshots = scheduler.export_hetinfer_prior_snapshots()
                self.assertEqual(len(snapshots), 1)
                self._assert_clean_complete_snapshot(
                    snapshots[0], scheduler, schedule, graph
                )

    def test_capture_toggle_same_scheduler_preserves_everything(self) -> None:
        for fixed in (False, True):
            with self.subTest(path="fixed" if fixed else "schedule"):
                scheduler = _FixtureBifocal(rand_seed=1234)
                initial_rng = scheduler._rng.getstate()

                scheduler.enable_hetinfer_prior_capture(False)
                disabled_schedule = _run(scheduler, _graph(), fixed=fixed)
                disabled_placement = dict(scheduler._node_placement)
                disabled_rng = scheduler._rng.getstate()
                self.assertEqual(scheduler.export_hetinfer_prior_snapshots(), [])

                # Same object, immediately repeated from the same scheduler RNG
                # baseline; only the observational capture toggle changes.
                scheduler._rng.setstate(initial_rng)
                scheduler.enable_hetinfer_prior_capture(True)
                enabled_schedule = _run(scheduler, _graph(), fixed=fixed)
                enabled_placement = dict(scheduler._node_placement)
                enabled_rng = scheduler._rng.getstate()

                self.assertEqual(
                    _schedule_signature(enabled_schedule),
                    _schedule_signature(disabled_schedule),
                )
                self.assertEqual(enabled_placement, disabled_placement)
                self.assertEqual(enabled_rng, disabled_rng)
                self.assertEqual(len(scheduler.export_hetinfer_prior_snapshots()), 1)
                self.assertEqual(
                    scheduler.export_hetinfer_prior_snapshots()[0][
                        "schedule_call_index"
                    ],
                    1,
                )

    def test_capture_handles_barrier_inputs_without_data_routes(self) -> None:
        graph = _graph()
        graph.barrier_edges = {("source", "left")}
        scheduler = _FixtureBifocal(rand_seed=123)
        scheduler.enable_hetinfer_prior_capture(True)
        _run(scheduler, graph, fixed=True)
        snapshot = scheduler.export_hetinfer_prior_snapshots()[0]
        barrier = next(
            entry
            for entry in snapshot["inputs"]
            if entry["consumer_op_id"] == "prefill:1:left"
            and entry["producer_op_id"] == "prefill:1:source"
        )
        self.assertEqual(barrier["semantics"], "barrier")
        self.assertEqual(barrier["bytes"], 0)
        self.assertEqual(barrier["source_residencies"], [])
        self.assertEqual(barrier["destination_devices"], [])
        self.assertFalse(
            any(
                route["tensor_id"] == "prefill:1:barrier:source->left"
                for route in snapshot["routes"]
            )
        )

    def test_failed_schedule_and_fixed_plan_publish_no_partial_snapshot(self) -> None:
        for fixed in (False, True):
            with self.subTest(path="fixed" if fixed else "schedule"):
                scheduler = _FixtureBifocal(rand_seed=3, fail_on_commit="left")
                scheduler.enable_hetinfer_prior_capture(True)
                with self.assertRaisesRegex(RuntimeError, "fixture commit failure"):
                    _run(scheduler, _graph(), fixed=fixed)
                self.assertEqual(scheduler.export_hetinfer_prior_snapshots(), [])

                scheduler._fixture_fail_on_commit = None
                _run(scheduler, _graph(), fixed=fixed)
                self.assertEqual(
                    [
                        snapshot["schedule_call_index"]
                        for snapshot in scheduler.export_hetinfer_prior_snapshots()
                    ],
                    [1],
                )

                # A later disabled run also consumes no public instance id;
                # the next successful captured run is exactly 2.
                scheduler.enable_hetinfer_prior_capture(False)
                _run(scheduler, _graph(), fixed=fixed)
                scheduler.enable_hetinfer_prior_capture(True)
                _run(scheduler, _graph(), fixed=fixed)
                self.assertEqual(
                    [
                        snapshot["schedule_call_index"]
                        for snapshot in scheduler.export_hetinfer_prior_snapshots()
                    ],
                    [1, 2],
                )

    def test_failed_finalization_publishes_no_partial_snapshot(self) -> None:
        for fixed in (False, True):
            with self.subTest(path="fixed" if fixed else "schedule"):
                graph = _graph()
                graph.nodes["source"].attrs["fixture_export_failure"] = True
                scheduler = _FixtureBifocal(rand_seed=11)
                scheduler.enable_hetinfer_prior_capture(True)
                with self.assertRaisesRegex(
                    RuntimeError, "fixture export finalization failure"
                ):
                    _run(scheduler, graph, fixed=fixed)
                self.assertEqual(scheduler.export_hetinfer_prior_snapshots(), [])

                graph.nodes["source"].attrs.pop("fixture_export_failure")
                _run(scheduler, graph, fixed=fixed)
                self.assertEqual(
                    scheduler.export_hetinfer_prior_snapshots()[0][
                        "schedule_call_index"
                    ],
                    1,
                )

    def test_external_input_rejects_empty_sources_and_negative_bytes(self) -> None:
        invalid_cases = (
            ({"source_devices": [], "bytes": 1, "layout": "ND"}, "cannot be empty"),
            (
                {"source_devices": ["NPU0"], "bytes": -1, "layout": "ND"},
                "cannot be negative",
            ),
        )
        for raw_input, expected_error in invalid_cases:
            with self.subTest(raw_input=raw_input):
                graph = _graph()
                graph.nodes["source"].attrs["hetinfer_external_inputs"] = [
                    {"tensor_id": "invalid", **raw_input}
                ]
                scheduler = _FixtureBifocal(rand_seed=17)
                scheduler.enable_hetinfer_prior_capture(True)
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    _run(scheduler, graph, fixed=True)
                self.assertEqual(scheduler.export_hetinfer_prior_snapshots(), [])

    def test_allreduce_exports_canonical_placement_and_all_output_residencies(
        self,
    ) -> None:
        graph = TaskGraph()
        for node_id in ("left", "right", "sink"):
            graph.add_node(
                TaskNode(
                    id=node_id,
                    name=node_id.upper(),
                    bytes_read=4096,
                    bytes_write=8192,
                    allowed={"cpu": False, "npu": True, "pim": False},
                    attrs={
                        "fixture_compute_s": {
                            "NPU0": 0.001,
                            "NPU1": 0.001,
                        }
                    },
                )
            )
        graph.add_node(
            TaskNode(
                id="allreduce",
                name="ALLREDUCE",
                bytes_read=8192,
                bytes_write=8192,
                allowed={"cpu": True, "npu": True, "pim": False},
                attrs={"primitive": "ALLREDUCE"},
            )
        )
        graph.add_edge("left", "allreduce")
        graph.add_edge("right", "allreduce")
        graph.add_edge("allreduce", "sink")

        scheduler = _FixtureBifocal(rand_seed=19, include_second_npu=True)
        scheduler.enable_hetinfer_prior_capture(True)
        schedule = scheduler.schedule_with_plan(
            graph,
            "prefill",
            {
                "order": ("left", "right", "allreduce", "sink"),
                "device_by_node": {
                    "left": "NPU0",
                    "right": "NPU1",
                    "sink": "NPU0",
                },
            },
        )

        task_by_node = {task.node_id: task for task in schedule}
        self.assertEqual(task_by_node["allreduce"].device, "COMM")
        self.assertEqual(scheduler._node_placement["allreduce"], "NPU0")
        self.assertEqual(
            scheduler._collective_output_devs["allreduce"], {"NPU0", "NPU1"}
        )

        snapshot = scheduler.export_hetinfer_prior_snapshots()[0]
        operator = next(
            item
            for item in snapshot["operators"]
            if item["op_id"] == "prefill:1:allreduce"
        )
        # ScheduledTask uses the public COMM marker, while the scheduler's
        # committed placement and the static prior use the canonical concrete
        # output device selected by the collective implementation.
        self.assertEqual(operator["expert_device"], "NPU0")
        self.assertEqual(operator["legal_devices"], ["NPU0"])
        self.assertGreater(operator["service_s"]["NPU0"], 0.0)
        self.assertAlmostEqual(
            operator["service_s"]["NPU0"],
            task_by_node["allreduce"].finish - task_by_node["allreduce"].start,
            places=12,
        )
        self.assertEqual(
            operator["atlas_descriptor"]["collective_primitive"], "ALLREDUCE"
        )
        self.assertEqual(
            operator["atlas_descriptor"]["collective_participants"],
            ["NPU0", "NPU1"],
        )

        context = next(
            item
            for item in snapshot["collective_contexts"]
            if item["op_id"] == "prefill:1:allreduce"
        )
        self.assertEqual(
            context,
            {
                "op_id": "prefill:1:allreduce",
                "primitive": "ALLREDUCE",
                "topology": "fc",
                "canonical_device_id": "NPU0",
                "participant_device_ids": ["NPU0", "NPU1"],
                "output_device_ids": ["NPU0", "NPU1"],
                "resource_device_ids": ["NPU0", "NPU1"],
                "tensor_bytes": 8192,
                "internal_transport": "included_in_t_service",
            },
        )

        staging_inputs = {
            entry["producer_op_id"]: entry
            for entry in snapshot["inputs"]
            if entry["consumer_op_id"] == "prefill:1:allreduce"
        }
        self.assertEqual(set(staging_inputs), {"prefill:1:left", "prefill:1:right"})
        self.assertEqual(
            staging_inputs["prefill:1:left"]["destination_devices"], ["NPU0"]
        )
        self.assertEqual(
            staging_inputs["prefill:1:right"]["destination_devices"], ["NPU1"]
        )
        self.assertEqual(
            {entry["semantics"] for entry in staging_inputs.values()},
            {"collective_staging"},
        )
        route_by_key = {
            (
                route["tensor_id"],
                route["source_device_id"],
                route["destination_device_id"],
            ): route
            for route in snapshot["routes"]
        }
        # The expert placement already owns each collective input at its fixed
        # participant.  These staging routes are zero; T_service above is only
        # the atomic ALLREDUCE transport and cannot include an extra input hop.
        for producer, participant in (("left", "NPU0"), ("right", "NPU1")):
            resident_stage = route_by_key[
                (f"prefill:1:tensor:{producer}", participant, participant)
            ]
            self.assertEqual(resident_stage["duration_s"], 0.0)
            self.assertFalse(resident_stage["requires_atlas"])

        # If an upstream tensor has already spilled to CPU, the same fixed
        # expert context remains valid and its staging cost is explicit.
        self.assertGreater(
            route_by_key[("prefill:1:tensor:left", "CPU0", "NPU0")][
                "duration_s"
            ],
            0.0,
        )
        self.assertGreater(
            route_by_key[("prefill:1:tensor:right", "CPU0", "NPU1")][
                "duration_s"
            ],
            0.0,
        )

        output_routes = {
            (
                route["source_device_id"],
                route["destination_device_id"],
                route["duration_s"],
            )
            for route in snapshot["routes"]
            if route["tensor_id"] == "prefill:1:tensor:allreduce"
        }
        self.assertEqual(
            {(source, destination) for source, destination, _ in output_routes},
            {
                (source, destination)
                for source in {"CPU0", "NPU0", "NPU1"}
                for destination in {"CPU0", "NPU0", "NPU1"}
            },
        )
        for source, destination, duration_s in output_routes:
            if source == destination:
                self.assertEqual(duration_s, 0.0)
            else:
                self.assertGreater(duration_s, 0.0)

    def test_pim_collective_service_and_nonresident_routes_require_atlas(
        self,
    ) -> None:
        graph = TaskGraph()
        for node_id in ("npu_input", "pim_input", "sink"):
            graph.add_node(
                _node(
                    node_id,
                    read_bytes=4096,
                    write_bytes=8192,
                    npu_compute_s=0.001,
                    pim_compute_s=0.001,
                )
            )
        graph.add_node(
            TaskNode(
                id="allreduce",
                name="ALLREDUCE",
                bytes_read=8192,
                bytes_write=8192,
                allowed={"cpu": True, "npu": True, "pim": True},
                attrs={"primitive": "ALLREDUCE"},
            )
        )
        graph.add_edge("npu_input", "allreduce")
        graph.add_edge("pim_input", "allreduce")
        graph.add_edge("allreduce", "sink")

        scheduler = _FixtureBifocal(rand_seed=23)
        scheduler.enable_hetinfer_prior_capture(True)
        scheduler.schedule_with_plan(
            graph,
            "prefill",
            {
                "order": ("npu_input", "pim_input", "allreduce", "sink"),
                "device_by_node": {
                    "npu_input": "NPU0",
                    "pim_input": "PIM0",
                    "sink": "NPU0",
                },
            },
        )
        snapshot = scheduler.export_hetinfer_prior_snapshots()[0]
        operator = next(
            item
            for item in snapshot["operators"]
            if item["op_id"] == "prefill:1:allreduce"
        )
        self.assertEqual(operator["legal_devices"], ["NPU0"])
        self.assertIsNone(operator["service_s"]["NPU0"])
        context = snapshot["collective_contexts"][0]
        self.assertEqual(context["participant_device_ids"], ["NPU0", "PIM0"])
        self.assertEqual(context["resource_device_ids"], ["NPU0", "PIM0"])

        pim_nonresident = [
            route
            for route in snapshot["routes"]
            if "PIM0"
            in {route["source_device_id"], route["destination_device_id"]}
            and route["source_device_id"] != route["destination_device_id"]
        ]
        self.assertTrue(pim_nonresident)
        for route in pim_nonresident:
            self.assertIsNone(route["duration_s"])
            self.assertTrue(route["requires_atlas"])

    def test_scatter_stages_on_host_and_exports_host_plus_target_outputs(
        self,
    ) -> None:
        graph = TaskGraph()
        for node_id in ("source", "sink"):
            graph.add_node(
                TaskNode(
                    id=node_id,
                    name=node_id.upper(),
                    bytes_read=4096,
                    bytes_write=4096,
                    allowed={"cpu": False, "npu": True, "pim": False},
                    attrs={
                        "fixture_compute_s": {
                            "NPU0": 0.001,
                            "NPU1": 0.001,
                        }
                    },
                )
            )
        graph.add_node(
            TaskNode(
                id="scatter",
                name="SCATTER",
                bytes_read=4096,
                bytes_write=4096,
                allowed={"cpu": True, "npu": True, "pim": False},
                attrs={
                    "primitive": "SCATTER",
                    "targets": ["NPU0", "NPU1"],
                    "scatter_mode": "broadcast",
                },
            )
        )
        graph.add_edge("source", "scatter")
        graph.add_edge("scatter", "sink")

        scheduler = _FixtureBifocal(rand_seed=29, include_second_npu=True)
        scheduler.enable_hetinfer_prior_capture(True)
        schedule = scheduler.schedule_with_plan(
            graph,
            "prefill",
            {
                "order": ("source", "scatter", "sink"),
                "device_by_node": {"source": "NPU0", "sink": "NPU1"},
            },
        )
        snapshot = scheduler.export_hetinfer_prior_snapshots()[0]
        context = next(
            item
            for item in snapshot["collective_contexts"]
            if item["op_id"] == "prefill:1:scatter"
        )
        self.assertEqual(context["primitive"], "SCATTER")
        self.assertEqual(context["canonical_device_id"], "CPU0")
        self.assertEqual(context["participant_device_ids"], ["CPU0"])
        self.assertEqual(context["output_device_ids"], ["CPU0", "NPU0", "NPU1"])
        self.assertEqual(context["resource_device_ids"], ["CPU0", "NPU0", "NPU1"])

        scatter_operator = next(
            item
            for item in snapshot["operators"]
            if item["op_id"] == "prefill:1:scatter"
        )
        scatter_task = next(task for task in schedule if task.node_id == "scatter")
        self.assertAlmostEqual(
            scatter_operator["service_s"]["CPU0"],
            scatter_task.finish - scatter_task.start,
            places=12,
        )

        staging = next(
            entry
            for entry in snapshot["inputs"]
            if entry["consumer_op_id"] == "prefill:1:scatter"
        )
        self.assertEqual(staging["semantics"], "collective_staging")
        self.assertEqual(staging["destination_devices"], ["CPU0"])

        output = next(
            entry
            for entry in snapshot["inputs"]
            if entry["producer_op_id"] == "prefill:1:scatter"
        )
        self.assertEqual(
            {item["device_id"] for item in output["source_residencies"]},
            {"CPU0", "NPU0", "NPU1"},
        )

    def test_transfer_binds_fixed_source_and_destination_context(self) -> None:
        graph = TaskGraph()
        for node_id in ("source", "sink"):
            graph.add_node(
                TaskNode(
                    id=node_id,
                    name=node_id.upper(),
                    bytes_read=4096,
                    bytes_write=4096,
                    allowed={"cpu": False, "npu": True, "pim": False},
                    attrs={
                        "fixture_compute_s": {
                            "NPU0": 0.001,
                            "NPU1": 0.001,
                        }
                    },
                )
            )
        graph.add_node(
            TaskNode(
                id="transfer",
                name="TRANSFER",
                bytes_read=4096,
                bytes_write=4096,
                allowed={"cpu": True, "npu": True, "pim": False},
                attrs={
                    "primitive": "TRANSFER",
                    "src": "NPU0",
                    "dst": "NPU1",
                    "bytes": 4096,
                },
            )
        )
        graph.add_edge("source", "transfer")
        graph.add_edge("transfer", "sink")

        scheduler = _FixtureBifocal(rand_seed=31, include_second_npu=True)
        scheduler.enable_hetinfer_prior_capture(True)
        schedule = scheduler.schedule_with_plan(
            graph,
            "prefill",
            {
                "order": ("source", "transfer", "sink"),
                "device_by_node": {"source": "NPU0", "sink": "NPU1"},
            },
        )
        snapshot = scheduler.export_hetinfer_prior_snapshots()[0]
        context = snapshot["collective_contexts"][0]
        self.assertEqual(context["primitive"], "TRANSFER")
        self.assertEqual(context["participant_device_ids"], ["NPU0"])
        self.assertEqual(context["output_device_ids"], ["NPU1"])
        self.assertEqual(context["canonical_device_id"], "NPU1")
        self.assertEqual(context["resource_device_ids"], ["NPU0", "NPU1"])

        staging = next(
            entry
            for entry in snapshot["inputs"]
            if entry["consumer_op_id"] == "prefill:1:transfer"
        )
        self.assertEqual(staging["destination_devices"], ["NPU0"])
        transfer_operator = next(
            entry
            for entry in snapshot["operators"]
            if entry["op_id"] == "prefill:1:transfer"
        )
        transfer_task = next(task for task in schedule if task.node_id == "transfer")
        self.assertAlmostEqual(
            transfer_operator["service_s"]["NPU1"],
            transfer_task.finish - transfer_task.start,
            places=12,
        )

    def test_real_cost_model_primitives_match_hand_calculation(self) -> None:
        cluster = Cluster()
        cluster.topology = "fc"
        cpu = DeviceSpec(
            name="CPU0",
            type="cpu",
            tflops=1.0,
            mem_bw_GBs=8.0,
            mem_capacity_GB=1.0,
        )
        npu = DeviceSpec(
            name="NPU0",
            type="npu",
            tflops=10.0,
            mem_bw_GBs=1000.0,
            mem_capacity_GB=1.0,
        )
        cluster.add_device(cpu)
        cluster.add_device(npu)
        link = LinkSpec(
            bw_GBs=2.0,
            latency_s=3e-6,
            overhead_s=5e-6,
            flit_size_B=16,
            max_payload_B=256,
        )
        cluster.connect("CPU0", "NPU0", link)
        cost = CostModel(
            cluster,
            dtype="fp16",
            pim_fast_mode=True,
            npu_backend="fast",
        )
        scheduler = BifocalScheduler(
            cluster=cluster,
            cost=cost,
            label=SimpleNamespace(
                kv_place="host",
                kv_in_pim=False,
                kv_total_bytes=0,
            ),
            batch=2,
            seq_len=3,
            buffer=None,
            rand_seed=1,
        )

        graph = TaskGraph()
        graph.add_node(
            TaskNode(
                id="k_write",
                name="K_WRITE",
                allowed={"cpu": True, "npu": False, "pim": False},
                attrs={"n_kv_heads": 2, "head_dim": 4},
            )
        )
        # B=2, Hkv=2, head_dim=4, T=3, fp16=2 B => 96 bytes.
        expected_write_bytes = 2 * 2 * 4 * 3 * 2
        _, actual_write_bytes = cost.estimate_activation_bytes(
            graph.nodes["k_write"], 2, 3, "prefill"
        )
        self.assertEqual(actual_write_bytes, expected_write_bytes)
        actual_kv_service = scheduler._hetinfer_compute_service_s(
            graph, "k_write", cpu, "prefill"
        )
        # The production host-KV execution path represents predecessor->host
        # as movement and commits no additional host-local store primitive.
        # Export must retain that exact zero instead of adding mem_time again.
        self.assertEqual(actual_kv_service, 0.0)

        scheduler.enable_hetinfer_prior_capture(True)
        schedule = scheduler.schedule_with_plan(
            graph,
            "prefill",
            {
                "order": ("k_write",),
                "device_by_node": {"k_write": "CPU0"},
            },
        )
        self.assertEqual(schedule[0].device, "CPU0")
        self.assertEqual(schedule[0].finish - schedule[0].start, 0.0)
        operator = scheduler.export_hetinfer_prior_snapshots()[0]["operators"][0]
        self.assertEqual(operator["expert_device"], "CPU0")
        self.assertEqual(operator["service_s"], {"CPU0": 0.0})

        bytes_amount = 513
        packets = 3
        bytes_on_wire = bytes_amount + packets * 16
        expected_route = 3e-6 + 5e-6 + bytes_on_wire / (2.0 * 1024.0**3)
        self.assertAlmostEqual(
            cost.comm_cost(cpu, npu, bytes_amount), expected_route, places=18
        )
        actual_route = scheduler._hetinfer_route_time_s(
            cpu, npu, bytes_amount, source_layout="ND"
        )
        self.assertAlmostEqual(actual_route, expected_route, places=18)


if __name__ == "__main__":
    unittest.main()
