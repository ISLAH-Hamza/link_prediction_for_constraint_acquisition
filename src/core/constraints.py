from .base import Constraint, Relation
from ortools.sat.python import cp_model
import operator as op

class EqualVal(Constraint):
    def __init__(self, scope, parameter) -> None:
        relation = Relation("==val", 1, False, parameter)
        super().__init__(scope, relation)
        self.v = self.scope[0].name
     

    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.eq(example[self.scope[0].name], self.relation.parameter)
    
    def to_ortools(self, model, variables) -> None:
        model.Add(variables[self.v] == self.relation.parameter)
    
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.v] == self.relation.parameter).OnlyEnforceIf(b)
        model.Add(variables[self.v] != self.relation.parameter).OnlyEnforceIf(b.Not())



class NotEqualVal(Constraint):
    def __init__(self, scope,parameter) -> None:
        relation = Relation("!=val", 1, False, parameter)
        super().__init__(scope, relation)
        self.v = self.scope[0].name
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.ne(example[self.scope[0].name], self.relation.parameter)
    
    def to_ortools(self, model, variables):
        model.Add(variables[self.v] != self.relation.parameter)
    
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.v] != self.relation.parameter).OnlyEnforceIf(b)
        model.Add(variables[self.v] == self.relation.parameter).OnlyEnforceIf(b.Not())



class GreaterEqualVal(Constraint):
    def __init__(self, scope,parameter) -> None:
        relation = Relation(">=val", 1, False, parameter)
        super().__init__(scope, relation)
        self.v = self.scope[0].name
     
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.ge(example[self.scope[0].name], self.relation.parameter)

    def to_ortools(self, model, variables):
        model.Add(variables[self.v] >= self.relation.parameter)
    
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.v] >= self.relation.parameter).OnlyEnforceIf(b)
        model.Add(variables[self.v] < self.relation.parameter).OnlyEnforceIf(b.Not())


class LessEqualVal(Constraint):
    def __init__(self, scope,parameter) -> None:
        relation = Relation("<=val", 1, False, parameter)
        super().__init__(scope, relation)
        self.v = self.scope[0].name
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.le(example[self.scope[0].name], self.relation.parameter)
    
    def to_ortools(self, model, variables):
        model.Add(variables[self.v] <= self.relation.parameter)
    
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.v] <= self.relation.parameter).OnlyEnforceIf(b)
        model.Add(variables[self.v] > self.relation.parameter).OnlyEnforceIf(b.Not())


class GreaterVal(Constraint):
    def __init__(self, scope,parameter) -> None:
        relation = Relation(">val", 1, False, parameter)
        super().__init__(scope, relation)
        self.v = self.scope[0].name
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.gt(example[self.scope[0].name], self.relation.parameter)
    
    def to_ortools(self, model, variables):
        model.Add(variables[self.v] > self.relation.parameter)
    
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.v] > self.relation.parameter).OnlyEnforceIf(b)
        model.Add(variables[self.v] <= self.relation.parameter).OnlyEnforceIf(b.Not())


class LessVal(Constraint):
    def __init__(self, scope,parameter) -> None:
        relation = Relation("<val", 1, False, parameter)
        super().__init__(scope, relation)
        self.v = self.scope[0].name
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.lt(example[self.scope[0].name], self.relation.parameter)
    
    def to_ortools(self, model, variables):
        model.Add(variables[self.v] < self.relation.parameter)
    
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.v] < self.relation.parameter).OnlyEnforceIf(b)
        model.Add(variables[self.v] >= self.relation.parameter).OnlyEnforceIf(b.Not())


class Equal(Constraint):
    def __init__(self, scope) -> None:
        relation = Relation("==", 2, False)
        super().__init__(scope, relation)
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.eq(example[self.scope[0].name], example[self.scope[1].name])

    def to_ortools(self, model, variables):
        model.Add(variables[self.scope[0].name] == variables[self.scope[1].name])
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.scope[0].name] == variables[self.scope[1].name]).OnlyEnforceIf(b)
        model.Add(variables[self.scope[0].name] != variables[self.scope[1].name]).OnlyEnforceIf(b.Not())


class NotEqual(Constraint):
    def __init__(self, scope) -> None:
        relation = Relation("!=", 2, False)
        super().__init__(scope, relation)
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.ne(example[self.scope[0].name], example[self.scope[1].name])
    
    def to_ortools(self, model, variables):
        model.Add(variables[self.scope[0].name] != variables[self.scope[1].name])
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.scope[0].name] != variables[self.scope[1].name]).OnlyEnforceIf(b)
        model.Add(variables[self.scope[0].name] == variables[self.scope[1].name]).OnlyEnforceIf(b.Not())


class GreaterEqual(Constraint):
    def __init__(self, scope) -> None:
        relation = Relation(">=", 2, True)
        super().__init__(scope, relation)
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.ge(example[self.scope[0].name], example[self.scope[1].name])
    
    def to_ortools(self, model, variables):
        model.Add(variables[self.scope[0].name] >= variables[self.scope[1].name])
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.scope[0].name] >= variables[self.scope[1].name]).OnlyEnforceIf(b)
        model.Add(variables[self.scope[0].name] < variables[self.scope[1].name]).OnlyEnforceIf(b.Not())


class LessEqual(Constraint):
    def __init__(self, scope) -> None:
        relation = Relation("<=", 2, True)
        super().__init__(scope, relation)
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.le(example[self.scope[0].name], example[self.scope[1].name])

    def to_ortools(self, model, variables):
        model.Add(variables[self.scope[0].name] <= variables[self.scope[1].name])
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.scope[0].name] <= variables[self.scope[1].name]).OnlyEnforceIf(b)
        model.Add(variables[self.scope[0].name] > variables[self.scope[1].name]).OnlyEnforceIf(b.Not())


class Greater(Constraint):
    def __init__(self, scope) -> None:
        relation = Relation(">", 2, True)
        super().__init__(scope, relation)

    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.gt(example[self.scope[0].name], example[self.scope[1].name])

    def to_ortools(self, model, variables):
        model.Add(variables[self.scope[0].name] > variables[self.scope[1].name])
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.scope[0].name] > variables[self.scope[1].name]).OnlyEnforceIf(b)
        model.Add(variables[self.scope[0].name] <= variables[self.scope[1].name]).OnlyEnforceIf(b.Not())


class Less(Constraint):
    def __init__(self, scope) -> None:
        relation = Relation("<", 2, True)
        super().__init__(scope, relation)
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.lt(example[self.scope[0].name], example[self.scope[1].name])
    def to_ortools(self, model, variables):
        model.Add(variables[self.scope[0].name] < variables[self.scope[1].name])
    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.scope[0].name] < variables[self.scope[1].name]).OnlyEnforceIf(b)
        model.Add(variables[self.scope[0].name] >= variables[self.scope[1].name]).OnlyEnforceIf(b.Not())


class DistEqual(Constraint):
    def __init__(self, scope, parameter) -> None:
        relation = Relation("||==val||", 2, False, parameter)
        super().__init__(scope, relation)
        self.v1 = self.scope[0].name
        self.v2 = self.scope[1].name

    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.eq(abs(example[self.v1] - example[self.v2]), self.relation.parameter)
    
    def to_ortools(self, model, variables):
        max_d = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_d, max_d, f"diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_d, f"abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable == self.relation.parameter)

    def to_ortools_boolean(self, b, model, variables):
        max_domain = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_domain, max_domain, f"boolean_diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_domain, f"boolean_abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable == self.relation.parameter).OnlyEnforceIf(b)
        model.Add(temp_variable != self.relation.parameter).OnlyEnforceIf(b.Not())


class DiffEqual(Constraint):
    """Signed difference: scope[0] - scope[1] == parameter.

    Directed (order matters), so it captures a one-sided offset such as
    "v1 is exactly `parameter` positions to the right of v2" in a single
    constraint, unlike the abs-based DistEqual which needs a companion
    Greater/Less to fix the direction.
    """
    def __init__(self, scope, parameter) -> None:
        relation = Relation("-==val", 2, True, parameter)
        super().__init__(scope, relation)
        self.v1 = self.scope[0].name
        self.v2 = self.scope[1].name

    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.eq(example[self.v1] - example[self.v2], self.relation.parameter)

    def to_ortools(self, model, variables):
        model.Add(variables[self.v1] - variables[self.v2] == self.relation.parameter)

    def to_ortools_boolean(self, b, model, variables):
        model.Add(variables[self.v1] - variables[self.v2] == self.relation.parameter).OnlyEnforceIf(b)
        model.Add(variables[self.v1] - variables[self.v2] != self.relation.parameter).OnlyEnforceIf(b.Not())


class DistNotEqual(Constraint):
    def __init__(self, scope, parameter) -> None:
        relation = Relation("||!=val||", 2, False, parameter)
        super().__init__(scope, relation)
        self.v1 = self.scope[0].name
        self.v2 = self.scope[1].name
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.ne(abs(example[self.v1]- example[self.v2]), self.relation.parameter)

    def to_ortools(self, model, variables):
        max_d = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_d, max_d, f"diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_d, f"abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable != self.relation.parameter)

    def to_ortools_boolean(self, b, model, variables):
        max_domain = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_domain, max_domain, f"boolean_diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_domain, f"boolean_abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable != self.relation.parameter).OnlyEnforceIf(b)
        model.Add(temp_variable == self.relation.parameter).OnlyEnforceIf(b.Not())



class DistGreater(Constraint):
    def __init__(self, scope, parameter) -> None:
        relation = Relation("||>val||", 2, True, parameter)
        super().__init__(scope, relation)
        self.v1 = self.scope[0].name
        self.v2 = self.scope[1].name

    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.gt(abs(example[self.v1] - example[self.v2]), self.relation.parameter)
    
    def to_ortools(self, model, variables):
        max_d = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_d, max_d, f"diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_d, f"abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable > self.relation.parameter)

    def to_ortools_boolean(self, b, model, variables):
        max_domain = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_domain, max_domain, f"boolean_diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_domain, f"boolean_abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable > self.relation.parameter).OnlyEnforceIf(b)
        model.Add(temp_variable <= self.relation.parameter).OnlyEnforceIf(b.Not())



class DistLess(Constraint):
    def __init__(self, scope, parameter) -> None:
        relation = Relation("||<val||", 2, True, parameter)
        super().__init__(scope, relation)
        self.v1 = self.scope[0].name
        self.v2 = self.scope[1].name
    
    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.lt(abs(example[self.v1] - example[self.v2]), self.relation.parameter)
    
    def to_ortools(self, model, variables):
        max_d = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_d, max_d, f"diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_d, f"abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable < self.relation.parameter)

    def to_ortools_boolean(self, b, model, variables):
        max_domain = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_domain, max_domain, f"boolean_diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_domain, f"boolean_abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable < self.relation.parameter).OnlyEnforceIf(b)
        model.Add(temp_variable >= self.relation.parameter).OnlyEnforceIf(b.Not())



class DistGreaterEqual(Constraint):
    def __init__(self, scope, parameter) -> None:
        relation = Relation("||>=val||", 2, True, parameter)
        super().__init__(scope, relation)
        self.v1 = self.scope[0].name
        self.v2 = self.scope[1].name

    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.ge(abs(example[self.v1] - example[self.v2]), self.relation.parameter)
    def to_ortools(self, model, variables):
        max_d = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_d, max_d, f"diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_d, f"abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable >= self.relation.parameter)

    def to_ortools_boolean(self, b, model, variables):
        max_domain = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_domain, max_domain, f"boolean_diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_domain, f"boolean_abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable >= self.relation.parameter).OnlyEnforceIf(b)
        model.Add(temp_variable < self.relation.parameter).OnlyEnforceIf(b.Not())



class DistLessEqual(Constraint):
    def __init__(self, scope, parameter) -> None:
        relation = Relation("||<=val||", 2, True, parameter)
        super().__init__(scope, relation)
        self.v1 = self.scope[0].name
        self.v2 = self.scope[1].name

    def check(self, example):
        if  all(v.name in example for v in self.scope):
            return op.le(abs(example[self.v1] - example[self.v2]), self.relation.parameter)

    def to_ortools(self, model, variables):
        max_d = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_d, max_d, f"diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_d, f"abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable <= self.relation.parameter)

    def to_ortools_boolean(self, b, model, variables):
        max_domain = max(self.scope[0].domain[1], self.scope[1].domain[1])
        diff = model.NewIntVar(-max_domain, max_domain, f"boolean_diff_{self.v1}_{self.v2}")
        model.Add(diff == variables[self.v1] - variables[self.v2])
        temp_variable = model.NewIntVar(0, max_domain, f"boolean_abs_{self.v1}_{self.v2}")
        model.AddAbsEquality(temp_variable, diff)
        model.Add(temp_variable <= self.relation.parameter).OnlyEnforceIf(b)
        model.Add(temp_variable > self.relation.parameter).OnlyEnforceIf(b.Not())



# Add here any additional constraints that you want to support in your model.
# Each constraint should be a subclass of the Constraint class and should implement the to_ortools
# and to_ortools_boolean methods to convert the constraint into a format that can be used by the OR-Tools solver.
