from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .schema import ShotTrainingRecord


@dataclass
class DomainSelectionV2232:
    session_id: str | None
    records: list[ShotTrainingRecord]
    engineering_records: list[ShotTrainingRecord]
    reason: str


def select_fresh_f2_domain(
    records: Sequence[ShotTrainingRecord], *, min_shots: int = 50
) -> DomainSelectionV2232:
    """Reserve the newest substantial F2/projector session as domain validation.

    It is excluded from challenger fitting. The set is allowed to gate research
    champion promotion, but it is NOT the protected holdout used for authority.
    """
    groups: dict[str, list[ShotTrainingRecord]] = {}
    for record in records:
        if record.source_kind != "f2_projected":
            continue
        groups.setdefault(record.session_id, []).append(record)
    eligible = [items for items in groups.values() if len(items) >= int(min_shots)]
    if not eligible:
        return DomainSelectionV2232(None, [], list(records), "no_f2_session_with_minimum_shots")
    latest = max(eligible, key=lambda items: max(float(r.timestamp) for r in items))
    sid = latest[0].session_id
    domain_ids = {id(r) for r in latest}
    engineering = [r for r in records if id(r) not in domain_ids]
    return DomainSelectionV2232(sid, list(latest), engineering, "latest_f2_session_reserved")
