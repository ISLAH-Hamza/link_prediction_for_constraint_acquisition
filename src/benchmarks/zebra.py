import random
from dataclasses import dataclass, field
from itertools import permutations, combinations
from core.base import Variable
from core.constraints import DiffEqual, DistEqual, Equal, EqualVal, Greater, NotEqual


@dataclass
class Model:
    """
    The Zebra puzzle (Einstein's Riddle) encoded as a CSP.

    25 variables represent five attributes (nationality, color, drink, pet,
    cigarette) across five houses. Each variable has domain [0, 4].
    Constraints enforce uniqueness within each attribute group and the
    logical clues of the puzzle.
    """

    domain: list[int] = field(default_factory=lambda: [0, 4])
    variables: list[Variable] = field(init=False)
    indexes: dict[str, Variable] = field(init=False)
    constraints: set = field(init=False)
    bais:set = field(init=False)

    def __post_init__(self) -> None:
        self.variables = [Variable(f"{i}", self.domain) for i in range(25)]
        random.shuffle(self.variables)
        self._init_indexes()
        self._setup_constraints()
        self.bais = self._Bais()



    def _init_indexes(self):
        v = self.variables
        self.indexes = {
            "Nor": v[0],  "En": v[1],    "Sp": v[2],    "Uk": v[3],    "Jp": v[4],
            "blue": v[5], "red": v[6],   "green": v[7], "yellow": v[8], "Ivre": v[9],
            "cofee": v[10], "milk": v[11], "tee": v[12], "jus": v[13], "water": v[14],
            "fox": v[15], "dog": v[16],  "snail": v[17], "hors": v[18], "zebra": v[19],
            "kool": v[20], "parl": v[21], "old": v[22],  "luck": v[23], "chest": v[24],
        }




    def _setup_constraints(self) -> None:
        
        idx = self.indexes
        v = self.variables
        self.constraints = {
            EqualVal([idx["Nor"]], 1),
            DistEqual([idx["Nor"], idx["blue"]], 1),
            EqualVal([idx["milk"]], 2),
            Equal([idx["En"], idx["red"]]),
            Equal([idx["cofee"], idx["green"]]),
            Equal([idx["kool"], idx["yellow"]]),
            DiffEqual([idx["Ivre"], idx["green"]], 1),
            Equal([idx["Sp"], idx["dog"]]),
            Equal([idx["Uk"], idx["tee"]]),
            Equal([idx["Jp"], idx["parl"]]),
            Equal([idx["old"], idx["snail"]]),
            Equal([idx["luck"], idx["jus"]]),
            DistEqual([idx["chest"], idx["fox"]], 1),
            DistEqual([idx["kool"], idx["hors"]], 1),
        }

        for group in (v[0:5], v[5:10], v[10:15], v[15:20], v[20:25]):
            self._add_alldiff(group)




    def _add_alldiff(self, variables: list[Variable]) -> None:
        """Add NotEqual constraints between every pair in a variable group.

        Args:
            variables: Group of variables that must all take distinct values.
        """
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self.constraints.add(NotEqual(scope=[variables[i], variables[j]]))



    def _Bais(self):
        bais = set()
        for var in self.variables:
            bais.add(EqualVal([var], 1))
            bais.add(EqualVal([var], 2))

        for template in (DistEqual, Equal, NotEqual, Greater):
            if template == Greater:
                for var1, var2 in permutations(self.variables, 2):
                    bais.add(template([var1, var2]))
            elif template == DiffEqual:
                for var1, var2 in permutations(self.variables, 2):
                    bais.add(template([var1, var2], parameter=1))
            else:
                for var1, var2 in combinations(self.variables, 2):
                    if template == DistEqual:
                            bais.add(template([var1, var2], parameter=1))
                    else:
                        bais.add(template([var1, var2]))

        return bais
    

if __name__ == "__main__":
    model = Model()
    print("number of variables:", len(model.variables))
    print("number of constraints:", len(model.constraints))
    print("number of bais:", len(model.bais))
    