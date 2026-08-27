from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.offline.media_bank_v219 import MediaAssetV219
from src.engine.offline.hole_patch_bank_v220 import HolePatchBankV220
from src.engine.offline.scenario_generator_v220 import OfflineScenarioGeneratorV220, ScenarioProfileV220, scenario_fingerprint


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _make_colour_test_asset(root: Path) -> MediaAssetV219:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    image[..., 0] = np.tile(np.linspace(30, 190, 320, dtype=np.uint8), (180, 1))
    image[..., 1] = np.tile(np.linspace(220, 60, 320, dtype=np.uint8), (180, 1))
    image[..., 2] = 80
    cv2.rectangle(image, (30, 35), (125, 120), (15, 25, 230), -1)
    cv2.circle(image, (235, 88), 34, (210, 190, 20), -1)
    cv2.putText(image, "RGB QA", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 2, cv2.LINE_AA)
    path = root / "colour_asset.png"
    cv2.imwrite(str(path), image)
    return MediaAssetV219(
        media_id="colour_asset",
        family_id="colour_asset_family",
        split="train",
        category="photo_or_image",
        kind="image",
        path=str(path.relative_to(root)),
        frame_count=1,
        width=320,
        height=180,
        license="CC0",
        source_url="selftest",
        content_sha256="na",
        perceptual_hash="na",
        metadata={},
    )


def _make_bank(root: Path) -> HolePatchBankV220:
    bank_dir = root / "content" / "ai" / "holes"
    bank_dir.mkdir(parents=True)
    patch = np.full((128, 128), 220, dtype=np.uint8)
    cv2.circle(patch, (64, 64), 4, 24, -1, cv2.LINE_AA)
    cv2.circle(patch, (64, 64), 7, 246, 1, cv2.LINE_AA)
    cv2.circle(patch, (66, 62), 2, 12, -1, cv2.LINE_AA)
    cv2.imwrite(str(bank_dir / "synt_0000001.png"), patch)
    (bank_dir / "synt_0000001.json").write_text(json.dumps({"image_type": "synt", "background_mode": "white", "session_id": "selftest", "patch_size": [128, 128]}), encoding="utf-8")
    return HolePatchBankV220.discover(bank_dir, root=root)


def _test_determinism_rgb_and_qc() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        asset = _make_colour_test_asset(root)
        bank = _make_bank(root)
        profile = ScenarioProfileV220(
            width=320,
            height=180,
            pre_frames=2,
            post_frames=2,
            old_hole_min=2,
            old_hole_max=2,
            sensor_noise_sigma=0.0,
            blur_sigma_max=0.0,
            frame_gain_jitter=0.0,
            camera_gain_jitter=0.0,
            camera_gamma_jitter=0.0,
            camera_channel_jitter=0.0,
            camera_black_level_jitter=0.0,
            use_camera_hole_patch_bank=True,
        )
        gen = OfflineScenarioGeneratorV220(profile=profile, media_assets=[asset], repo_root=root, hole_bank=bank)
        a = gen.generate(1234)
        b = gen.generate(1234)
        _assert(a.spec.to_dict() == b.spec.to_dict(), "scenario spec is not deterministic")
        _assert(scenario_fingerprint(a.spec) == scenario_fingerprint(b.spec), "scenario fingerprint is unstable")
        _assert(all(np.array_equal(x, y) for x, y in zip(a.pre_frames_rgb, b.pre_frames_rgb)), "RGB PRE frames are not deterministic")
        _assert(all(np.array_equal(x, y) for x, y in zip(a.post_frames_rgb, b.post_frames_rgb)), "RGB POST frames are not deterministic")
        rgb = a.pre_frames_rgb[-1]
        _assert(not np.all(rgb[..., 0] == rgb[..., 1]), "observed output collapsed to grayscale")
        qa = a.spec.metadata.get("qa", {})
        _assert(float(qa.get("local_mean_abs_diff", 0.0)) >= profile.qa_local_diff_min, "local hole diff is too weak")
        _assert(float(qa.get("center_mean_darkening", 0.0)) >= profile.qa_center_darkening_min, "hole does not darken the centre")


def _test_static_scene_has_low_global_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        asset = _make_colour_test_asset(root)
        bank = _make_bank(root)
        profile = ScenarioProfileV220(
            width=320,
            height=180,
            pre_frames=2,
            post_frames=2,
            old_hole_min=0,
            old_hole_max=0,
            sensor_noise_sigma=0.0,
            blur_sigma_max=0.0,
            frame_gain_jitter=0.0,
            camera_gain_jitter=0.0,
            camera_gamma_jitter=0.0,
            camera_channel_jitter=0.0,
            camera_black_level_jitter=0.0,
            use_camera_hole_patch_bank=True,
        )
        gen = OfflineScenarioGeneratorV220(profile=profile, media_assets=[asset], repo_root=root, hole_bank=bank)
        s = gen.generate(9)
        qa = s.spec.metadata.get("qa", {})
        _assert(float(qa.get("static_global_mae", 999.0)) <= 0.2, "static scene still has too much PRE/POST global drift")


def _test_dynamic_background_tagging() -> None:
    profile = ScenarioProfileV220(width=320, height=180, pre_frames=3, post_frames=3, old_hole_min=1, old_hole_max=1, sensor_noise_sigma=0.0, blur_sigma_max=0.0)
    gen = OfflineScenarioGeneratorV220(profile=profile, media_assets=[], repo_root=Path("."), hole_bank=HolePatchBankV220([]))
    found = None
    for seed in range(1, 250):
        scenario = gen.generate(seed)
        if "dynamic_background" in scenario.spec.challenge_tags:
            found = scenario
            break
    _assert(found is not None, "could not generate a dynamic procedural challenge")
    _assert(found.spec.gt_camera_xy[0] >= 0 and found.spec.gt_camera_xy[1] >= 0, "GT coordinates missing")


def main() -> int:
    print("V2.20 SELFTEST")
    print("==============")
    _test_determinism_rgb_and_qc(); print("[PASS] deterministic RGB observed output + compact visible hole QA")
    _test_static_scene_has_low_global_drift(); print("[PASS] static image scenarios keep PRE/POST camera state stable")
    _test_dynamic_background_tagging(); print("[PASS] dynamic procedural scenarios still preserve labelled world semantics")
    print("\nAll V2.20 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
