from src.engine.camera.object_physical_query import ObjectPhysicalQuery, MAX_FRAMES


def candidate(object_id, x):
    return {
        "confirmed": True,
        "evidence_score": 5.0,
        "camera_xy": [x, 100],
        "regions": [{
            "object_id": object_id,
            "inside_exact_shape": True,
        }],
    }


def row(object_id, x):
    return {
        "object_id": object_id,
        "candidates": [candidate(object_id, x)],
    }


def query(rows):
    q = ObjectPhysicalQuery.__new__(ObjectPhysicalQuery)
    q.per_object = rows
    q.object_hit_requested = True
    q.frames = 2
    q.peak_ts = 10.0
    return q


q = query([row("soda_can", 100), row("gold_can", 200)])

assert set(q._supported_objects_current_frame()) == {
    "soda_can", "gold_can"
}
assert q._await_object_disambiguation(10.10) is True

q.per_object = [row("soda_can", 100)]
assert q._await_object_disambiguation(10.10) is False

q.per_object = []
assert q._await_object_disambiguation(10.10) is True

q.per_object = [row("soda_can", 100), row("gold_can", 200)]
q.frames = MAX_FRAMES
assert q._await_object_disambiguation(10.10) is False

q.frames = 2
assert q._await_object_disambiguation(10.21) is False

q.object_hit_requested = False
assert q._await_object_disambiguation(10.10) is False

print("PASS: OBJECT_HIT current-frame ambiguity continuation")
