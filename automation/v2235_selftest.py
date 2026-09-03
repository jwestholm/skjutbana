from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np

from src.engine.ai.training_v223.evidence_patch_v2235 import (
    EVIDENCE_CHANNELS, EVIDENCE_PATCH_SIZE, PATCH_POSITIVE_RADIUS_PX,
    PATCH_NEGATIVE_RADIUS_PX, extract_evidence_patches,
)
from src.engine.ai.training_v223.evidence_model_v2235 import (
    EVIDENCE_VECTOR_SIZE, EvidenceModelV2235, SampledEvidenceShotV2235,
    _init_model, evidence_vector, train_stage,
)

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f'[PASS] {msg}')

def main() -> int:
    print('V2.23.5 SELFTEST\n================')
    channels = np.zeros((EVIDENCE_CHANNELS, 81, 81), np.uint8)
    for c in range(EVIDENCE_CHANNELS):
        channels[c, 38:43, 38:43] = np.uint8(180 + min(c*8, 70))
    patches = extract_evidence_patches(channels, np.asarray([[40,40],[12,12]], np.float32))
    check(patches.shape == (2, EVIDENCE_CHANNELS, EVIDENCE_PATCH_SIZE, EVIDENCE_PATCH_SIZE), 'registered evidence patch geometry is stable')
    check(float(patches[0,:,4,4].mean()) > float(patches[1,:,4,4].mean()), 'candidate-centred evidence signal survives pooling')
    vec = evidence_vector(patches)
    check(vec.shape == (2, EVIDENCE_VECTOR_SIZE), 'evidence patch vector preserves spatial features')
    check(PATCH_POSITIVE_RADIUS_PX == 6.0 and PATCH_NEGATIVE_RADIUS_PX == 42.0, 'tight positive / wide neutral label contract is frozen')

    # Dependency-free learnability proof: central persistent event should outrank
    # structured but off-centre false evidence after pairwise training.
    rng = np.random.default_rng(2350)
    samples = []
    for shot in range(16):
        npos, nneg = 8, 40
        pos = rng.integers(0, 18, size=(npos,EVIDENCE_CHANNELS,9,9), dtype=np.uint8)
        neg = rng.integers(0, 35, size=(nneg,EVIDENCE_CHANNELS,9,9), dtype=np.uint8)
        pos[:,2,3:6,3:6] += 180
        pos[:,4,3:6,3:6] += 160
        pos[:,6,3:6,3:6] += 190
        pos[:,7,3:6,3:6] += 150
        # hard negatives have strong evidence, but displaced from candidate centre
        neg[:,2,0:3,0:3] += 150
        neg[:,6,6:9,6:9] += 170
        pp = np.clip(pos,0,255).astype(np.uint8)
        nn = np.clip(neg,0,255).astype(np.uint8)
        distances = np.concatenate([np.zeros(npos,np.float32), np.full(nneg,80.0,np.float32)])
        samples.append(SampledEvidenceShotV2235('s',str(shot),np.concatenate([pp,nn]),distances))
    model = _init_model('mlp', hidden=32, seed=2351)
    train_stage(model, samples, epochs=18, learning_rate=0.005, l2=0.0005, seed=2352, stage_name='selftest')
    test_pos = np.zeros((12,EVIDENCE_CHANNELS,9,9),np.uint8); test_neg = np.zeros_like(test_pos)
    test_pos[:,2,3:6,3:6]=220; test_pos[:,4,3:6,3:6]=200; test_pos[:,6,3:6,3:6]=230; test_pos[:,7,3:6,3:6]=190
    test_neg[:,2,0:3,0:3]=220; test_neg[:,6,6:9,6:9]=230
    check(float(model.score_patches(test_pos).mean()) > float(model.score_patches(test_neg).mean()) + 0.15, 'evidence learner can learn centred NEW-hole structure')

    with tempfile.TemporaryDirectory() as td:
        model.save(td)
        loaded = EvidenceModelV2235.load(td)
        check(np.allclose(model.score_patches(test_pos), loaded.score_patches(test_pos), atol=1e-5), 'model save/load is deterministic and allow_pickle-free')
    check(model.metadata.get('live_authority', False) is False, 'V2.23.5 model remains shadow-only')
    print('\nAll V2.23.5 selftests passed.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
