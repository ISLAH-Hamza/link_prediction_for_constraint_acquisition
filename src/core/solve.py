from core.base import *
from ortools.sat.python import cp_model


def Solve(L,vars):
    """
        Solve the constraints in L using the given variables in cp_model from ortools.
        args:
            L       : The list of constraints to solve.
            vars    : The list of variables to use in the constraints.
            logger  : An optional logger to log information.
    """
    
    m=cp_model.CpModel()
    variables={v.name:m.NewIntVar(*v.domain,v.name) for v in vars}

    for conj in L:
        if isinstance(conj, Constraint): conj.to_ortools(m, variables=variables)
        else:
            for c in conj: c.to_ortools(m, variables=variables)

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    status = solver.Solve(m)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        sol={i: solver.Value(variables[i])for i in variables.keys()}
        return sol




class _SolutionCollector(cp_model.CpSolverSolutionCallback):
    """Collects solutions up to a given limit, then stops search."""

    def __init__(self, var_map: dict, limit: int = 100) -> None:
        super().__init__()
        self._var_map = var_map
        self._limit = limit
        self.solutions: list[dict] = []

    def on_solution_callback(self) -> None:
        sol = {name: self.Value(v) for name, v in self._var_map.items()}
        self.solutions.append(sol)
        if len(self.solutions) >= self._limit:
            self.StopSearch()


def _enumerate(constraints: list, variables: list) -> list[dict]:
    """Enumerate up to 100 solutions of a constraint list."""
    m = cp_model.CpModel()
    var_map = {v.name: m.NewIntVar(*v.domain, v.name) for v in variables}
    for conj in constraints:
        if isinstance(conj, Constraint):
            conj.to_ortools(m, variables=var_map)
        else:
            for c in conj:
                c.to_ortools(m, variables=var_map)
    collector = _SolutionCollector(var_map, limit=100)
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_workers = 1
    solver.Solve(m, collector)
    return collector.solutions


def _satisfies(sol: dict, constraints: list) -> bool:
    """Return True iff sol satisfies every constraint (or conjunction) in the list."""
    for conj in constraints:
        if isinstance(conj, Constraint):
            if conj.check(sol) == False:
                return False
        else:
            if any(c.check(sol) == False for c in conj):
                return False
    return True


def equivalent(Target: list, Learned: list, variables: list) -> dict:
    """
    Check both directions of equivalence between Target and Learned.

    Soundness  (Learned -> Target): every solution of Learned satisfies Target.
    Completeness (Target -> Learned): every solution of Target satisfies Learned.

    Args:
        Target:    List of target constraints (or TargetNetwork.constraints).
        Learned:   List of learned constraints (or conjunctions).
        variables: List of Variable objects defining the search space.

    Returns:
        True if Learned and Target are equivalent (same solution set, up to 100 samples).
    """
    learned_solutions = _enumerate(Learned, variables)
    target_solutions = _enumerate(Target, variables)

    spurious = [s for s in learned_solutions if not _satisfies(s, Target)]
    missing  = [s for s in target_solutions  if not _satisfies(s, Learned)]

    return len(spurious) == 0 and len(missing) == 0