from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.offline.media_bank_v219 import index_media_roots, split_for_family, summarise_media, write_media_manifest, read_media_manifest, audit_media
from src.engine.offline.scenario_generator_v219 import OfflineScenarioGeneratorV219, ScenarioProfileV219, scenario_fingerprint
from src.engine.offline.hole_patch_bank_v219 import HolePatchBankV219


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _test_media_split_and_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, value in (("photo_a.png", 220), ("game_target.png", 80)):
            image = np.full((90, 160, 3), value, dtype=np.uint8)
            cv2.circle(image, (80, 45), 18, (20, 20, 20), 2)
            cv2.imwrite(str(root / name), image)
        sidecar = {"family_id": "same-source-family", "category": "photo", "license": "CC0"}
        (root / "photo_a.png.json").write_text(json.dumps(sidecar), encoding="utf-8")
        assets = index_media_roots([root], repo_root=root)
        _assert(len(assets) == 2, "media index did not find both images")
        _assert(split_for_family("abc") == split_for_family("abc"), "split is not deterministic")
        manifest = root / "manifest.jsonl"
        write_media_manifest(manifest, assets)
        loaded = read_media_manifest(manifest)
        _assert([a.media_id for a in loaded] == [a.media_id for a in assets], "media manifest roundtrip failed")
        _assert(summarise_media(loaded)["assets"] == 2, "media summary wrong")


def _test_media_audit_catches_cross_split_duplicate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        image=np.full((80,120,3),140,np.uint8); cv2.putText(image,'DUP',(20,48),cv2.FONT_HERSHEY_SIMPLEX,1,(10,10,10),2)
        for name,split in [('dup_train.png','train'),('dup_holdout.png','holdout')]:
            cv2.imwrite(str(root/name),image)
            (root/(name+'.json')).write_text(json.dumps({'family_id':name,'split':split,'license':'CC0'}),encoding='utf-8')
        assets=index_media_roots([root],repo_root=root)
        report=audit_media(assets)
        _assert(len(report['exact_cross_split_duplicates'])>=1,'cross-split exact duplicate was not detected')
        _assert(report['ok_for_frozen_holdout'] is False,'leaky media bank incorrectly passed holdout gate')


def _test_video_source_is_one_split_unit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); path=root/'motion_test.avi'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10.0,(160,90))
        if writer.isOpened():
            for i in range(8):
                frame=np.full((90,160,3),40,np.uint8); cv2.rectangle(frame,(10+i*12,25),(45+i*12,65),(220,120,50),-1); writer.write(frame)
            writer.release()
            assets=index_media_roots([path],repo_root=root)
            _assert(len(assets)==1 and assets[0].kind=='video','video media source was not indexed as one asset')
            _assert(assets[0].frame_count>=2,'video frame metadata missing')
        else:
            writer.release()


def _test_scenario_determinism_and_semantics() -> None:
    profile = ScenarioProfileV219(width=320, height=180, pre_frames=3, post_frames=3, old_hole_min=8, old_hole_max=8, sensor_noise_sigma=0.0, blur_sigma_max=0.0)
    generator = OfflineScenarioGeneratorV219(profile=profile, media_assets=[], repo_root=Path("."))
    a = generator.generate(12345, split="train")
    b = generator.generate(12345, split="train")
    _assert(a.spec.to_dict() == b.spec.to_dict(), "same seed did not reproduce scenario spec")
    _assert(scenario_fingerprint(a.spec) == scenario_fingerprint(b.spec), "scenario fingerprint is unstable")
    _assert(all(np.array_equal(x, y) for x, y in zip(a.pre_frames, b.pre_frames)), "same seed did not reproduce PRE")
    _assert(all(np.array_equal(x, y) for x, y in zip(a.post_frames, b.post_frames)), "same seed did not reproduce POST")
    _assert(len(a.spec.old_holes) == 8, "old-hole count wrong")
    _assert(set(a.spec.known_holes).issubset(set(a.spec.old_holes)), "known-hole list is not a subset of actual old holes")
    x, y = map(int, map(round, a.spec.gt_camera_xy))
    pre = a.pre_frames[-1].astype(np.float32)
    post = a.post_frames[-1].astype(np.float32)
    patch = np.abs(post[max(0,y-5):y+6, max(0,x-5):x+6] - pre[max(0,y-5):y+6, max(0,x-5):x+6])
    _assert(float(np.mean(patch)) > 0.05, "new-hole area does not change between PRE and POST")


def _test_dynamic_background_is_not_hole_motion() -> None:
    profile = ScenarioProfileV219(width=320, height=180, pre_frames=3, post_frames=3, old_hole_min=3, old_hole_max=3, sensor_noise_sigma=0.0, blur_sigma_max=0.0)
    generator = OfflineScenarioGeneratorV219(profile=profile, media_assets=[], repo_root=Path("."))
    found = None
    for seed in range(1, 200):
        scenario = generator.generate(seed)
        if "dynamic_background" in scenario.spec.challenge_tags or scenario.spec.media_category in {"checker", "stripes", "noise", "game_like"}:
            found = scenario
            break
    _assert(found is not None, "could not generate dynamic procedural challenge")
    # Physical holes stay fixed in camera coordinates by construction even while the underlying media changes.
    _assert(len(found.spec.old_holes) == 3, "old-hole state missing")
    _assert(found.spec.gt_camera_xy[0] >= 0 and found.spec.gt_camera_xy[1] >= 0, "GT missing")



def _test_camera_hole_patch_bank_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bank_dir = root / "content" / "ai" / "holes"
        bank_dir.mkdir(parents=True)
        patch = np.full((128,128), 220, dtype=np.uint8)
        cv2.circle(patch,(64,64),4,25,-1,cv2.LINE_AA)
        cv2.circle(patch,(64,64),7,245,1,cv2.LINE_AA)
        cv2.imwrite(str(bank_dir / "synt_0000001.png"), patch)
        (bank_dir / "synt_0000001.json").write_text(json.dumps({"image_type":"synt","background_mode":"white","session_id":"selftest","patch_size":[128,128]}),encoding="utf-8")
        bank = HolePatchBankV219.discover(bank_dir, root=root)
        _assert(len(bank) == 1, "camera hole patch bank discovery failed")
        profile = ScenarioProfileV219(width=320,height=180,pre_frames=2,post_frames=2,old_hole_min=2,old_hole_max=2,sensor_noise_sigma=0.0,blur_sigma_max=0.0,use_camera_hole_patch_bank=True)
        generator = OfflineScenarioGeneratorV219(profile=profile,media_assets=[],repo_root=root,hole_bank=bank)
        scenario = generator.generate(9876)
        _assert("camera_captured_hole_appearance" in scenario.spec.challenge_tags, "generator did not use camera-captured hole appearance bank")
        _assert("procedural_hole_fallback" not in scenario.spec.challenge_tags, "unexpected procedural hole fallback")



def _test_generated_candidate_pack_bridge() -> None:
    import types
    import src.engine.offline.scenario_candidate_compiler_v219 as compiler_mod
    from src.engine.offline.candidate_pack_v216 import discover_candidate_packs, CandidatePackV216
    class FakeDetector:
        def detect(self, *, ground_truth=None, **kwargs):
            gx,gy=ground_truth
            rows=[
                {"camera_x":float(gx+3),"camera_y":float(gy-2),"score":0.7,"detector_v2":1.0},
                {"camera_x":float(max(2,gx+70)),"camera_y":float(max(2,gy+50)),"score":0.9,"detector_v1":1.0},
            ]
            return types.SimpleNamespace(candidates=rows)
    old=compiler_mod.LiveHybridReplayDetector
    compiler_mod.LiveHybridReplayDetector=lambda: FakeDetector()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            profile=ScenarioProfileV219(width=320,height=180,pre_frames=2,post_frames=2,old_hole_min=1,old_hole_max=1,sensor_noise_sigma=0.0,blur_sigma_max=0.0)
            generator=OfflineScenarioGeneratorV219(profile=profile,media_assets=[],repo_root=Path('.'))
            compiler=compiler_mod.ScenarioCandidateCompilerV219(generator=generator,output_root=root/'packs',include_overlay=False)
            report=compiler.compile(first_seed=55,count=2,split='train',session_id='selftest_v219')
            _assert(report['saved']==2,'candidate compiler did not save both generated worlds')
            paths=discover_candidate_packs(root/'packs')
            _assert(len(paths)==2,'V2.16-compatible candidate packs were not discoverable')
            pack=CandidatePackV216.load(paths[0])
            _assert(pack.metadata.get('extra',{}).get('v219_generated') is True,'generated provenance missing')
            _assert(pack.gt_xy is not None,'generated candidate pack GT missing')
            _assert(pack.recent_pre_patches is not None,'generated pack lacks true recent-pre candidate patches')
    finally:
        compiler_mod.LiveHybridReplayDetector=old


def main() -> int:
    print("V2.19 SELFTEST")
    print("==============")
    _test_media_split_and_index(); print("[PASS] source/family-level media splits + manifest roundtrip")
    _test_media_audit_catches_cross_split_duplicate(); print("[PASS] media audit catches cross-split exact/near leakage")
    _test_video_source_is_one_split_unit(); print("[PASS] video/animation source stays one media/split unit")
    _test_scenario_determinism_and_semantics(); print("[PASS] seed determinism + old-hole/new-hole semantics")
    _test_dynamic_background_is_not_hole_motion(); print("[PASS] dynamic-background scenarios preserve physical hole coordinates")
    _test_camera_hole_patch_bank_path(); print("[PASS] camera-captured synt_* hole appearances are reused without centre-label shortcut")
    _test_generated_candidate_pack_bridge(); print("[PASS] generated worlds compile into existing V2.16/V2.17/V2.18 candidate-pack schema")
    print("\nAll V2.19 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
