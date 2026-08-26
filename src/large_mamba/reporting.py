from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    GO = "GO — useful retrofit"
    MECHANISTIC_ONLY = "MECHANISTIC ONLY"
    NO_GO = "NO-GO for brief retrofit"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class PairedSeedResult:
    seed: int
    ord_stress_auc: float
    sphere_stress_auc: float
    ord_easy_accuracy: float
    sphere_easy_accuracy: float
    ord_language_nll: float
    sphere_language_nll: float
    mechanism_improved: bool
    completed: bool = True
    numerically_stable: bool = True

    @property
    def stress_difference(self) -> float:
        return self.sphere_stress_auc - self.ord_stress_auc


def decide(results: list[PairedSeedResult], performance_parity: bool = True) -> Decision:
    pass
    if len(results) != 3 or {row.seed for row in results} != {0, 1, 2}:
        return Decision.INCONCLUSIVE
    if any(not row.completed or not row.numerically_stable for row in results):
        return Decision.INCONCLUSIVE
    easy_ok = all(row.sphere_easy_accuracy >= row.ord_easy_accuracy - 0.02 for row in results)
    language_ok = all(row.sphere_language_nll <= row.ord_language_nll * 1.03 for row in results)
    if not easy_ok or not language_ok:
        return Decision.INCONCLUSIVE
    differences = [row.stress_difference for row in results]
    utility = sum(differences) / 3 >= 0.05 and all(value > 0 for value in differences)
    mechanism = any(row.mechanism_improved for row in results)
    if utility and mechanism:
        return Decision.GO
    if not utility and performance_parity and all(row.mechanism_improved for row in results):
        return Decision.MECHANISTIC_ONLY
    return Decision.NO_GO
