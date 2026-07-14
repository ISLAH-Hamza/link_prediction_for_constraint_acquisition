import itertools
import operator as _op
from abc import ABC, abstractmethod


class Variable:
    """
        A class to represent a variable with a name and a numeric domain.
    """
    def __init__(self, name, domain) -> None:
        # input validation
        assert isinstance(name, str), "The identifier should be a string"
        assert isinstance(domain, (list, tuple)), "Domain must be a list or tuple"
        assert len(domain) == 2, "The domain should contain exactly two elements"
        assert all(isinstance(x, (int, float)) for x in domain), "The domain should contain numeric values"
        lo, hi = domain
        assert lo < hi, "The first element of the domain should be less than the second"

        self.name = name
        self.domain = [lo, hi]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Variable):
            return NotImplemented
        return (self.name, tuple(self.domain)) == (other.name, tuple(other.domain))

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.domain)))

    def __str__(self) -> str:
        return f"name:{self.name} | domain:{self.domain}"




class Relation:
    """
        A class to represent a relation with an operator, arity, directionality, and optional parameter.
    """
    def __init__(self, operator, arity, directed, parameter=None) -> None:
      
        self.operator = operator
        self.arity = arity
        self.directed = directed
        self.parameter = parameter

    def __eq__(self, rel):
            return (self.operator, self.parameter) == (rel.operator, rel.parameter)
    
    def __hash__(self):
        # Avoid hashing None directly — its hash is based on id() and
        # therefore non-deterministic across processes.
        p = self.parameter if self.parameter is not None else ""
        return hash((self.operator, p))
       
    def __str__(self) -> str:
        if self.parameter is None:
            return f"relation: {self.operator}"
        else:
            return f"relation: {self.operator} | params: {self.parameter} "







class Constraint(ABC):
    """
    Abstract base class to represent a constraint in the model.
    """

    def __init__(self, scope, relation) -> None:
        self.scope = scope
        self.relation = relation

    def __eq__(self, C) -> bool:
        if self.relation.directed:
            return (self.relation, self.scope) == (C.relation, C.scope)
        else:
            return (self.relation, frozenset(self.scope)) == (C.relation, frozenset(C.scope))

    def __hash__(self) -> int:
        if self.relation.directed:
            return hash((self.relation, tuple(self.scope)))
        else:
            return hash((self.relation, frozenset(self.scope)))

    def __str__(self) -> str:
        scope = ' && '.join([str(v) for v in self.scope])
        return f"{self.relation} ## {scope}"

    @abstractmethod
    def check(self, example):
        """
        Check whether the constraint is satisfied by the example.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def to_ortools(self,model, variables):
        """
        Convert the constraint to an OR-Tools constraint and add it to the model.
        Must be implemented by subclasses."""
        pass

    @abstractmethod
    def to_ortools_boolean(self, b, model, variables):
        """
        Associate the constraint with a boolean variable in the OR-Tools model.
        Must be implemented by subclasses.
        """
        pass




class TargetNetwork:
    """
        A class to represent the target network (the user's oracle/answers).
    """
    def __init__(self, constraints=None, mode="default") -> None:
        self.constraints = constraints
        self.req_yes = 0
        self.req_no = 0
        self.ask_counter = 0
        self.mode = mode
        self.max_time = 0

        if mode == "default":
            if constraints is None:
                raise ValueError("Please provide constraints for the default mode")

    def ask(self, example):
        self.ask_counter += 1
        if self.mode == "human":
            print("Is the example valid? (yes/no):")
            print(f"Example: {example}")
            return input().strip().lower() == "yes"
        else:
            return all(c.check(example) is not False for c in self.constraints)


    def ask_req(self, c):

        self.ask_counter += 1
        if c in self.constraints:
            self.req_yes += 1
            return True
        else:
            self.req_no += 1
            return False
        
    def set_time(self, time):
        if time > self.max_time:
            self.max_time = time


    def get_statistics(self):
        return {
            "query": self.ask_counter,
            "ask":  self.ask_counter - (self.req_yes + self.req_no),
            "ask_req": self.req_yes + self.req_no,
            "yes": self.req_yes,
            "no": self.req_no,
            "time": self.max_time
        }

    def __str__(self) -> str:
        return  f"Total constraints: {len(self.constraints)}\nTotal asks: {self.ask_counter}"
