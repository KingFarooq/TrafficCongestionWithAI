"""
test_restrictions.py  —  P1.4 verification artifact

Fixed scenario: the full SOLUTIONS catalogue (30 candidates).  For each
implemented restriction predicate we assert:
        1  <=  number of candidates removed  <=  expected maximum

Run:   pytest -v test_restrictions.py
Table: python test_restrictions.py        (prints the paper-ready results table)
"""
import pytest

from image_processing_pipeline import (
    SOLUTIONS, RESTRICTIONS, RESTRICTION_PREDICATES, IMPLEMENTED_RESTRICTIONS,
    filter_solutions,
)

ALL = [s for cat in SOLUTIONS.values() for s in cat]

# restriction -> (predicate type, expected removed on the 30-item catalogue)
SPEC = {
    "no new lanes":             ("name keyword (add + lane)",   5),
    "no construction":          ("attribute (physical, cost>0.4)", 11),
    "no overpasses":            ("name keyword",                2),
    "no underpasses":           ("name keyword",                1),
    "no bike lanes":            ("name keyword",                2),
    "no bus lanes":             ("name keyword",                1),
    "no HOV lanes":             ("name keyword",                1),
    "no signal timing changes": ("name keyword",                1),
    "no pedestrian signals":    ("name keyword",                1),
}


def removed_by(restriction):
    kept = {s["name"] for s in filter_solutions(ALL, [restriction])}
    return [s["name"] for s in ALL if s["name"] not in kept]


def test_spec_covers_registry():
    assert set(SPEC) == set(RESTRICTION_PREDICATES)
    assert set(IMPLEMENTED_RESTRICTIONS) == set(RESTRICTION_PREDICATES)


def test_registry_is_subset_of_catalogue():
    assert set(RESTRICTION_PREDICATES) <= set(RESTRICTIONS)
    assert len(RESTRICTIONS) == 48


@pytest.mark.parametrize("restriction", sorted(SPEC))
def test_predicate_removes_between_1_and_expected(restriction):
    _, expected = SPEC[restriction]
    n = len(removed_by(restriction))
    assert 1 <= n <= expected, f"{restriction}: removed {n}, expected 1..{expected}"
    assert n == expected, f"{restriction}: removed {n}, catalogue count is {expected}"


def test_hov_removes_exactly_the_hov_conversion():          # P1.1 regression
    assert removed_by("no HOV lanes") == ["Convert to HOV lane"]


def test_old_hov_predicate_was_a_noop():                    # documents the bug
    old = lambda s: "HOV" in s["name"].lower()
    assert not any(old(s) for s in ALL)


def test_unimplemented_restriction_is_ignored_not_crashing():
    assert filter_solutions(ALL, ["no tree removal"]) == ALL


def test_combined_restrictions_union():
    combo = ["no new lanes", "no construction"]
    union = set(removed_by(combo[0])) | set(removed_by(combo[1]))
    kept  = {s["name"] for s in filter_solutions(ALL, combo)}
    assert {s["name"] for s in ALL} - kept == union


# ── paper-ready table ────────────────────────────────────────
if __name__ == "__main__":
    print(f"{'Restriction':<26}{'Predicate type':<34}{'Removed':>8}{'Expected':>9}  Result")
    print("-" * 88)
    for r in IMPLEMENTED_RESTRICTIONS:
        ptype, expected = SPEC[r]
        n = len(removed_by(r))
        ok = "PASS" if 1 <= n <= expected else "FAIL"
        print(f"{r:<26}{ptype:<34}{n:>8}{expected:>9}  {ok}")
