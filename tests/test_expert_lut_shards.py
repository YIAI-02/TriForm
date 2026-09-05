from types import SimpleNamespace

import pytest

import hetinfer_experiment_export as exporter
from hetinfer_camc_profile_export import build_expert_service_lut


@pytest.mark.parametrize("phase,maximum", [("prefill", 17), ("decode", 8)])
def test_shards_reproduce_complete_pointwise_lut(monkeypatch, tmp_path, phase, maximum):
    cost = SimpleNamespace(cluster=SimpleNamespace(devices={device: device for device in exporter.DEVICES}))
    node = SimpleNamespace(attrs={"dim": 16})
    sampled = []

    def service(cost, node, device, batch, sequence, phase):
        n_e = sequence if phase == "prefill" else batch
        sampled.append((device, n_e))
        return n_e * (1e-6 if device == exporter.NPU else 2e-6)

    monkeypatch.setattr(exporter, "_service", service)
    family = "ffn_w1"
    expected = exporter._expert_lut(cost, {family: node}, phase, maximum,
                                    tmp_path / "full.json")[family]
    sampled.clear()
    parts = [exporter._expert_lut(cost, {family: node}, phase, maximum,
             tmp_path / f"shard{shard}.json", shard, 4)[family] for shard in range(4)]
    npu, pim = {}, {device: {} for device in exporter.DEVICES[1:]}
    for part in parts:
        npu.update(part["npu"])
        for device in pim:
            pim[device].update(part["pim"][device])
    actual = build_expert_service_lut(max_tokens=maximum, activation_bytes_per_token=32,
                                    npu_anchors={exporter.NPU: npu}, pim_measurements=pim)
    assert actual == expected
    for device in pim:
        points = [n for d, n in sampled if d == device]
        assert sorted(points) == list(range(1, maximum + 1))
