from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations, permutations

from core.base import Variable
from core.constraints import Equal, NotEqual, DistGreater


@dataclass
class Model:
    """
    Radio Link Frequency Assignment Problem (RFLAP) benchmark encoded as a CSP.

    25 variables with domain [0, 32] represent radio links. Constraints enforce
    all-different within each of five groups and distance-greater-than interference
    constraints between adjacent groups.
    """

    domain: list[int] = field(default_factory=lambda: [0, 32])
    variables: list[Variable] = field(init=False)
    constraints: set = field(init=False)
    bais: set = field(init=False)

    def __post_init__(self) -> None:
        self.variables = [Variable(f"{i}", self.domain) for i in range(25)]
        random.shuffle(self.variables)
        self._setup_constraints()
        self.bais = self._Bais()

    def _setup_constraints(self) -> None:
        v = self.variables
        self.constraints = set()

        # All-different within each group of 5
        for group in (v[0:5], v[5:10], v[10:15], v[15:20], v[20:25]):
            self._add_alldiff(group)

        # Distance > 2 between adjacent groups
        self._add_dist_greater(v[0:5], v[5:10], parameter=2)
        self._add_dist_greater(v[5:10], v[10:15], parameter=2)
        self._add_dist_greater(v[15:20], v[20:25], parameter=2)

    def _add_alldiff(self, variables: list[Variable]) -> None:
        """Add NotEqual constraints between every pair in a variable group."""
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self.constraints.add(NotEqual(scope=[variables[i], variables[j]]))

    def _add_dist_greater(self, g1: list[Variable], g2: list[Variable], parameter: int) -> None:
        """Add DistGreater constraints between all pairs across two groups."""
        for v1 in g1:
            for v2 in g2:
                self.constraints.add(DistGreater(scope=[v1, v2], parameter=parameter))

    def _Bais(self) -> set:
        """Build the full candidate constraint set from the language."""
        bais = set()
        for v1, v2 in combinations(self.variables, 2):
            bais.add(Equal([v1, v2]))
            bais.add(NotEqual([v1, v2]))
            for p in range(1, 10):
                bais.add(DistGreater([v1, v2], parameter=p))
        
        return bais
    

if __name__ == "__main__":
    model = Model()
    print("number of variables:", len(model.variables))
    print("number of constraints:", len(model.constraints))
    print("number of bais:", len(model.bais))
    