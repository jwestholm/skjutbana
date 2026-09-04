from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np

from src.engine.ai.training_v223.heatmap_v2236 import (
    HEATMAP_CHANNELS, HEATMAP_STRIDE, coarse_to_camera, camera_to_coarse, downsample_block_mean,
)
from src.engine.ai.training_v223.heatmap_model_v2236 import (
    KERNEL_SIZE, HeatmapModelV2236, HeatmapTrainingSampleV2236,
    extract_grid_patches, init_heatmap_model, peak_camera_xy, train_stage,
)


def check(cond, msg):
    if not cond: raise AssertionError(msg)
    print(f'[PASS] {msg}')


def main() -> int:
    print('V2.23.6 SELFTEST\n================')
    full = np.zeros((HEATMAP_CHANNELS, 80, 120), np.uint8)
    full[2, 38:43, 58:63] = 240
    full[4, 39:42, 59:62] = 220
    full[6, 38:43, 58:63] = 250
    coarse = downsample_block_mean(full)
    check(coarse.shape == (HEATMAP_CHANNELS, 20, 30), 'registered full-frame evidence downsamples deterministically')
    cam = coarse_to_camera(15, 10); back = camera_to_coarse(*cam)
    check(abs(back[0]-15) < 1e-6 and abs(back[1]-10) < 1e-6, 'coarse/camera coordinate transform round-trips')

    # Direct localisation learnability: each shot has one centred multi-channel
    # event and several strong structured distractors elsewhere. No candidate list.
    rng = np.random.default_rng(2360)
    samples = []
    eval_maps = []
    gt_points = []
    for s in range(24):
        maps = rng.integers(0, 18, size=(HEATMAP_CHANNELS, 36, 48), dtype=np.uint8)
        gx = 8 + (s * 7) % 31; gy = 7 + (s * 5) % 22
        maps[2, gy-1:gy+2, gx-1:gx+2] = 230
        maps[4, gy-1:gy+2, gx-1:gx+2] = 210
        maps[6, gy-1:gy+2, gx-1:gx+2] = 245
        maps[7, gy-1:gy+2, gx-1:gx+2] = 205
        # false maxima: strong but one-sided/channel-incomplete structure
        for k in range(8):
            fx = int(rng.integers(3,45)); fy = int(rng.integers(3,33))
            maps[k % HEATMAP_CHANNELS, fy-1:fy+2, fx-1:fx+2] = 235
        pos = np.asarray([[gx, gy]], np.float32)
        neg = np.asarray([[3+(j*11+s)%43, 3+(j*7+2*s)%30] for j in range(45)], np.float32)
        pts = np.concatenate([pos, neg], axis=0)
        patches = extract_grid_patches(maps, pts)
        dist = np.concatenate([np.zeros(1,np.float32), np.full(len(neg),80.0,np.float32)])
        samples.append(HeatmapTrainingSampleV2236('s', str(s), patches, dist))
        eval_maps.append(maps); gt_points.append((gx,gy))
    model = init_heatmap_model('spatial_conv', hidden=8, seed=2361)
    train_stage(model, samples, epochs=28, learning_rate=0.0045, l2=0.0005, seed=2362, stage_name='selftest')
    good = 0
    for maps, (gx,gy) in zip(eval_maps, gt_points):
        peak = peak_camera_xy(model.score_map(maps), top_k=1)
        target = coarse_to_camera(gx,gy)
        if len(peak) and np.hypot(peak[0,0]-target[0], peak[0,1]-target[1]) <= 8.0:
            good += 1
    check(good >= 20, 'fully-convolutional heatmap learner localises synthetic NEW-hole evidence')
    with tempfile.TemporaryDirectory() as td:
        model.save(td); loaded = HeatmapModelV2236.load(td)
        a = model.score_map(eval_maps[0]); b = loaded.score_map(eval_maps[0])
        check(np.allclose(a,b,atol=1e-5), 'heatmap model save/load is deterministic and allow_pickle-free')
    check(model.metadata.get('live_authority', False) is False, 'V2.23.6 remains shadow-only')
    print('\nAll V2.23.6 selftests passed.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
