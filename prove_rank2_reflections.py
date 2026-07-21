#!/usr/bin/env python3
"""Exhaustively classify diagonal fermionic-rank-2 reflections.

For commuting fermionic number operators n_i, n_i^2 = n_i, an operator

    P = c0 + sum_i a_i n_i + sum_{i<j} b_ij n_i n_j

is diagonal in the occupation-number basis.  Thus P^2 = 1 exactly when the
associated multilinear polynomial takes values in {-1, +1} on {0, 1}^m.

This script enumerates every such normalized polynomial (c0 = 1), removes
dummy indices, quotients by index permutations, and checks the ten claimed
representatives.  It uses only the Python standard library.

Completeness of the finite search does not require guessing coefficient
ranges.  Values at 0, at the unit vectors, and at vectors e_i + e_j give

    c0 = P(0),
    a_i = P(e_i) - c0,
    b_ij = P(e_i+e_j) - c0 - a_i - a_j.

After fixing the harmless global sign by c0 = 1, choosing the +/-1 values on
the weight-one and weight-two inputs determines all coefficients uniquely.
There are only 2^(m + binomial(m, 2)) such choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Iterable, Iterator


Pair = tuple[int, int]


@dataclass(frozen=True)
class Polynomial:
    """A normalized quadratic multilinear polynomial (constant term 1)."""

    linear: tuple[int, ...]
    quadratic: tuple[int, ...]  # lexicographic pairs from combinations(range(m), 2)

    @property
    def arity(self) -> int:
        return len(self.linear)

    @property
    def pairs(self) -> tuple[Pair, ...]:
        return tuple(combinations(range(self.arity), 2))

    def value(self, occupation: tuple[int, ...]) -> int:
        return (
            1
            + sum(a * x for a, x in zip(self.linear, occupation))
            + sum(
                b * occupation[i] * occupation[j]
                for b, (i, j) in zip(self.quadratic, self.pairs)
            )
        )

    def is_reflection(self) -> bool:
        return all(
            self.value(x) in (-1, 1)
            for x in product((0, 1), repeat=self.arity)
        )

    def used_indices(self) -> frozenset[int]:
        used = {i for i, a in enumerate(self.linear) if a != 0}
        for b, (i, j) in zip(self.quadratic, self.pairs):
            if b != 0:
                used.update((i, j))
        return frozenset(used)

    def is_essential(self) -> bool:
        """True if no displayed index is a dummy (irrelevant) index."""
        return len(self.used_indices()) == self.arity

    def relabeled(self, old_index_at_new_position: tuple[int, ...]) -> "Polynomial":
        old_b = dict(zip(self.pairs, self.quadratic))
        new_a = tuple(self.linear[i] for i in old_index_at_new_position)
        new_b = tuple(
            old_b[tuple(sorted((old_index_at_new_position[i], old_index_at_new_position[j])))]
            for i, j in self.pairs
        )
        return Polynomial(new_a, new_b)

    def canonical(self) -> "Polynomial":
        """Lexicographically least representative under all index relabelings."""
        return min(
            (self.relabeled(p) for p in permutations(range(self.arity))),
            key=lambda q: (q.linear, q.quadratic),
        )


def polynomial_from_low_weight_values(
    m: int, singleton_values: Iterable[int], pair_values: Iterable[int]
) -> Polynomial:
    """Möbius-invert values at Hamming weights 0, 1, and 2."""
    linear = tuple(v - 1 for v in singleton_values)
    quadratic = tuple(
        v - 1 - linear[i] - linear[j]
        for v, (i, j) in zip(pair_values, combinations(range(m), 2))
    )
    return Polynomial(linear, quadratic)


def exhaustive_solutions(m: int) -> Iterator[Polynomial]:
    """Generate every normalized quadratic reflection on m named indices."""
    number_of_pairs = m * (m - 1) // 2
    for signs in product((-1, 1), repeat=m + number_of_pairs):
        p = polynomial_from_low_weight_values(m, signs[:m], signs[m:])
        if p.is_reflection():
            yield p


def make_polynomial(
    m: int, linear: dict[int, int], quadratic: dict[Pair, int]
) -> Polynomial:
    """Convenience constructor used to transcribe the claimed list."""
    return Polynomial(
        tuple(linear.get(i, 0) for i in range(m)),
        tuple(quadratic.get((i, j), 0) for i, j in combinations(range(m), 2)),
    )


def claimed_representatives() -> list[Polynomial]:
    """The ten polynomials p_1,...,p_10 in the question (indices start at 0)."""
    return [
        make_polynomial(1, {0: -2}, {}),
        make_polynomial(2, {}, {(0, 1): -2}),
        make_polynomial(2, {0: -2}, {(0, 1): 2}),
        make_polynomial(2, {0: -2, 1: -2}, {(0, 1): 2}),
        make_polynomial(2, {0: -2, 1: -2}, {(0, 1): 4}),
        make_polynomial(3, {0: -2}, {(1, 2): -2, (0, 2): 2}),
        make_polynomial(3, {0: -2}, {(1, 2): -2, (0, 2): 2, (0, 1): 2}),
        make_polynomial(3, {0: -2, 1: -2}, {(0, 2): 2, (0, 1): 2}),
        make_polynomial(
            3,
            {0: -2, 1: -2, 2: -2},
            {(0, 1): 2, (0, 2): 2, (1, 2): 2},
        ),
        make_polynomial(
            4,
            {0: -2, 1: -2},
            {(2, 3): -2, (0, 2): 2, (1, 3): 2, (0, 1): 2},
        ),
    ]


def main() -> None:
    claimed = claimed_representatives()
    claimed_by_arity = {
        m: {p.canonical() for p in claimed if p.arity == m} for m in range(1, 6)
    }

    print("m  candidates  solutions  essential  essential-orbits")
    all_coefficients: set[int] = set()
    for m in range(1, 6):
        candidates = 1 << (m + m * (m - 1) // 2)
        solutions = list(exhaustive_solutions(m))
        essential = [p for p in solutions if p.is_essential()]
        orbits = {p.canonical() for p in essential}
        for p in solutions:
            all_coefficients.update(p.linear)
            all_coefficients.update(p.quadratic)

        print(
            f"{m}  {candidates:10d}  {len(solutions):9d}  "
            f"{len(essential):9d}  {len(orbits):16d}"
        )
        assert orbits == claimed_by_arity[m], (
            f"classification mismatch at arity {m}",
            orbits - claimed_by_arity[m],
            claimed_by_arity[m] - orbits,
        )

    # These ranges are consequences of the low-weight +/-1 values, not search
    # assumptions.  In particular, -2 really can occur in a quadratic term.
    linear_coefficients = {
        a for m in range(1, 6) for p in exhaustive_solutions(m) for a in p.linear
    }
    quadratic_coefficients = {
        b for m in range(2, 6) for p in exhaustive_solutions(m) for b in p.quadratic
    }
    assert linear_coefficients == {-2, 0}
    assert quadratic_coefficients == {-2, 0, 2, 4}
    assert all(p.is_reflection() and p.is_essential() for p in claimed)

    print("\nPASS: the ten claimed representatives are exactly the essential orbits.")
    print("PASS: there is no solution depending essentially on five indices.")
    print(f"Observed linear coefficients:    {sorted(linear_coefficients)}")
    print(f"Observed quadratic coefficients: {sorted(quadratic_coefficients)}")


if __name__ == "__main__":
    main()
