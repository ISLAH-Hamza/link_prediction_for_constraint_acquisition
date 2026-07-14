from __future__ import annotations

"""
One solution to this jigsaw puzzle:
1 2 3 4 5 6
4 5 6 1 2 3
6 3 1 5 4 2
2 6 5 3 1 4
4 1 2 6 3 5
3 5 4 2 6 1
"""

import random
from dataclasses import dataclass, field
from itertools import combinations, permutations

from core.base import Variable
from core.constraints import GreaterEqual, LessEqual, NotEqual, Equal


@dataclass
class Model:
    """
    Jigsaw puzzle benchmark encoded as a CSP.

    36 variables with domain [1, 6] represent cells in a 6x6 grid.
    Constraints enforce all-different within each row and column, plus
    additional piece-adjacency inequalities defined by the jigsaw shape.
    """

    domain: list[int] = field(default_factory=lambda: [1, 6])
    variables: list[Variable] = field(init=False)
    constraints: set = field(init=False)
    bais: set = field(init=False)

    def __post_init__(self) -> None:
        self.variables = [Variable(f"{i}", self.domain) for i in range(36)]
        random.shuffle(self.variables)
        self._setup_constraints()
        self.bais = self._Bais()

    def _setup_constraints(self) -> None:
        v = self.variables
        self.constraints = {
            # Jigsaw shape adjacency constraints
            NotEqual([v[0],  v[13]]),
            NotEqual([v[6],  v[13]]),
            NotEqual([v[18], v[13]]),
            NotEqual([v[24], v[13]]),

            NotEqual([v[1],  v[8]]),
            NotEqual([v[1],  v[14]]),
            NotEqual([v[1],  v[15]]),
            NotEqual([v[2],  v[7]]),
            NotEqual([v[2],  v[15]]),
            NotEqual([v[7],  v[14]]),
            NotEqual([v[7],  v[15]]),
            NotEqual([v[8],  v[15]]),

            NotEqual([v[3],  v[10]]),
            NotEqual([v[3],  v[11]]),
            NotEqual([v[4],  v[9]]),
            NotEqual([v[4],  v[11]]),
            NotEqual([v[5],  v[9]]),
            NotEqual([v[5],  v[10]]),

            NotEqual([v[16], v[19]]),
            NotEqual([v[16], v[20]]),
            NotEqual([v[16], v[21]]),
            NotEqual([v[16], v[26]]),
            NotEqual([v[19], v[26]]),
            NotEqual([v[21], v[26]]),
            NotEqual([v[22], v[26]]),

            NotEqual([v[17], v[28]]),
            NotEqual([v[17], v[34]]),
            NotEqual([v[23], v[28]]),
            NotEqual([v[23], v[34]]),
            NotEqual([v[29], v[34]]),
            NotEqual([v[28], v[35]]),

            NotEqual([v[25], v[30]]),
            NotEqual([v[25], v[32]]),
            NotEqual([v[25], v[33]]),
            NotEqual([v[27], v[30]]),
            NotEqual([v[27], v[31]]),
            NotEqual([v[27], v[32]]),
        }
        # Row all-different (6 rows of 6)
        for start in (0, 6, 12, 18, 24, 30):
            self._add_alldiff_range(start, start + 5)
        # Column all-different (6 columns, step=6)
        for start in (0, 1, 2, 3, 4, 5):
            self._add_alldiff_range(start, start + 30)

    def _add_alldiff_range(self, start: int, end: int) -> None:
        """Add NotEqual constraints between variables at evenly spaced positions.

        Computes step as (end - start) // 5 so that exactly 6 variables are
        selected from index start to end inclusive.

        Args:
            start: First variable index.
            end:   Last variable index (inclusive).
        """
        step = (end - start) // 5
        indices = list(range(start, end + 1, step))
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                self.constraints.add(
                    NotEqual(scope=[self.variables[indices[i]], self.variables[indices[j]]])
                )

    def _Bais(self) -> set:
        """Build the full candidate constraint set from the language.

        Returns:
            Set of all constraints in the bias.
        """
        bais = set()
        for v1, v2 in combinations(self.variables, 2):
            bais.add(Equal([v1, v2]))
            bais.add(NotEqual([v1, v2]))
        for v1, v2 in permutations(self.variables, 2):
            bais.add(GreaterEqual([v1, v2]))
            bais.add(LessEqual([v1, v2]))
        return bais


if __name__ == "__main__":
    model = Model()
    print("number of variables:", len(model.variables))
    print("number of constraints:", len(model.constraints))
    print("number of bais:", len(model.bais))
    