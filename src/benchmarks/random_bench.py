from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations, permutations

from core.base import Variable
from core.constraints import Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual


# Map operator strings to constraint classes and a callable for validation
_CONSTRAINT_MAP = {
    "==": (Equal, lambda a, b: a == b),
    "!=": (NotEqual, lambda a, b: a != b),
    ">":  (Greater, lambda a, b: a > b),
    "<":  (Less, lambda a, b: a < b),
    ">=": (GreaterEqual, lambda a, b: a >= b),
    "<=": (LessEqual, lambda a, b: a <= b),
}


@dataclass
class Model:
    """
    Randomly generated benchmark with a planted (guaranteed) solution.

    20 variables with domain [0, 20]. Random binary constraints are added
    between variable pairs only if the planted solution satisfies them,
    controlled by a density parameter.
    """

    domain: list[int] = field(default_factory=lambda: [0, 20])
    n_variables: int = 20
    density: float = 0.4
    variables: list[Variable] = field(init=False)
    constraints: set = field(init=False)
    bais: set = field(init=False)

    def __post_init__(self) -> None:
        self.variables = [Variable(f"{i}", self.domain) for i in range(self.n_variables)]
        random.shuffle(self.variables)
        self._plant_solution()
        self._setup_constraints()
        self.bais = self._Bais()

    def _plant_solution(self) -> None:
        """Generate a random assignment for every variable."""
        self.planted_values = {}
        for var in self.variables:
            self.planted_values[var] = random.randint(self.domain[0], self.domain[1])

    def _setup_constraints(self) -> None:
        """Add random constraints that are satisfied by the planted solution."""
        self.constraints = set()
        ops = list(_CONSTRAINT_MAP.keys())

        for i in range(self.n_variables):
            for j in range(i + 1, self.n_variables):
                if random.random() < self.density:
                    var_a = self.variables[i]
                    var_b = self.variables[j]
                    val_a = self.planted_values[var_a]
                    val_b = self.planted_values[var_b]

                    rel = random.choice(ops)
                    cls, check = _CONSTRAINT_MAP[rel]

                    if check(val_a, val_b):
                        self.constraints.add(cls(scope=[var_a, var_b]))

    def _Bais(self) -> set:
        """Build the full candidate constraint set from the language."""
        bais = set()
        for v1, v2 in combinations(self.variables, 2):
            bais.add(Equal([v1, v2]))
            bais.add(NotEqual([v1, v2]))
        for v1, v2 in permutations(self.variables, 2):
            bais.add(Greater([v1, v2]))
            bais.add(Less([v1, v2]))
            bais.add(GreaterEqual([v1, v2]))
            bais.add(LessEqual([v1, v2]))
        return bais
    


if __name__ == "__main__":
    model = Model()
    print("number of variables:", len(model.variables))
    print("number of constraints:", len(model.constraints))
    print("number of bais:", len(model.bais))