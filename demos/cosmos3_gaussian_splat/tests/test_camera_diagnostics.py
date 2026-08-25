import json
from pathlib import Path

import numpy as np

from cosmos3_gsplat.camera_diagnostics import SHALLOW_YAW_ROT6D, _action_metrics


def test_shallow_arc_actions_remain_inside_reference_envelope() -> None:
    reference = np.asarray(
        json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "cookbooks/cosmos3/generator/action/assets/actions/camera_action.json"
            ).read_text()
        ),
        dtype=np.float32,
    )
    shallow = reference.copy()
    shallow[:, 3:] = SHALLOW_YAW_ROT6D
    outside = (shallow < reference.min(axis=0)) | (shallow > reference.max(axis=0))
    metrics = _action_metrics(shallow)
    assert not outside.any()
    np.testing.assert_array_equal(shallow[:, :3], reference[:, :3])
    assert 14.9 < metrics["rotation_deg_total"] < 15.1
    assert 0.249 < metrics["rotation_deg_mean"] < 0.251
