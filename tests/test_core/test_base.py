from __future__ import annotations

import pytest

from core.base import Constraint, Relation, TargetNetwork, Variable
from core.constraints import DistEqual, Equal, EqualVal, Greater, Less, NotEqual


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def var_x() -> Variable:
    """A simple variable named 'x' with domain [0, 10]."""
    return Variable("x", [0, 10])


@pytest.fixture
def var_y() -> Variable:
    """A simple variable named 'y' with domain [0, 10]."""
    return Variable("y", [0, 10])


@pytest.fixture
def var_z() -> Variable:
    """A simple variable named 'z' with domain [0, 10]."""
    return Variable("z", [0, 10])


# ===========================================================================
# TestVariable
# ===========================================================================


class TestVariableValidConstruction:
    """Happy-path construction of Variable."""

    def test_accepts_list_domain(self) -> None:
        """Variable constructs without error when domain is a list."""
        v = Variable("x", [0, 10])
        assert v.name == "x", f"Expected name 'x', got {v.name!r}"
        assert v.domain == [0, 10], f"Expected [0, 10], got {v.domain}"

    def test_accepts_tuple_domain(self) -> None:
        """Variable constructs without error when domain is a tuple."""
        v = Variable("x", (0, 10))
        assert v.domain == [0, 10], f"Expected [0, 10], got {v.domain}"

    def test_accepts_float_bounds(self) -> None:
        """Variable accepts float domain boundaries."""
        v = Variable("x", [0.0, 1.5])
        assert v.domain == [0.0, 1.5], f"Expected [0.0, 1.5], got {v.domain}"

    def test_accepts_negative_lower_bound(self) -> None:
        """Variable accepts a negative lower bound."""
        v = Variable("x", [-10, 0])
        assert v.domain == [-10, 0], f"Expected [-10, 0], got {v.domain}"

    def test_domain_stored_as_list(self) -> None:
        """Domain is always stored as a list even when a tuple is passed."""
        v = Variable("x", (1, 2))
        assert isinstance(v.domain, list), f"Expected list, got {type(v.domain)}"


class TestVariableInvalidName:
    """Variable raises AssertionError for non-string names."""

    @pytest.mark.parametrize("bad_name", [1, 1.5, None, [], ("x",)])
    def test_non_string_name_raises(self, bad_name: object) -> None:
        """Non-string name raises AssertionError."""
        with pytest.raises(AssertionError):
            Variable(bad_name, [0, 10])


class TestVariableInvalidDomain:
    """Variable raises AssertionError for invalid domain arguments."""

    @pytest.mark.parametrize("bad_domain", ["01", 0, None, {0, 10}])
    def test_non_list_or_tuple_domain_raises(self, bad_domain: object) -> None:
        """Domain that is neither list nor tuple raises AssertionError."""
        with pytest.raises(AssertionError):
            Variable("x", bad_domain)

    @pytest.mark.parametrize("bad_domain", [[], [0], [0, 5, 10], ()])
    def test_wrong_length_domain_raises(self, bad_domain: object) -> None:
        """Domain that does not have exactly two elements raises AssertionError."""
        with pytest.raises(AssertionError):
            Variable("x", bad_domain)

    @pytest.mark.parametrize("bad_domain", [["a", 10], [0, "b"], [None, 10]])
    def test_non_numeric_domain_raises(self, bad_domain: object) -> None:
        """Domain containing non-numeric values raises AssertionError."""
        with pytest.raises(AssertionError):
            Variable("x", bad_domain)

    @pytest.mark.parametrize("bad_domain", [[10, 0], [5, 5], [10, -1]])
    def test_lo_not_less_than_hi_raises(self, bad_domain: list) -> None:
        """Domain where lo >= hi raises AssertionError."""
        with pytest.raises(AssertionError):
            Variable("x", bad_domain)


class TestVariableStr:
    """Variable.__str__ produces the expected string."""

    def test_str_format(self) -> None:
        """__str__ returns 'name:<name> | domain:<domain>'."""
        v = Variable("x", [0, 10])
        assert str(v) == "name:x | domain:[0, 10]", f"Unexpected __str__: {str(v)!r}"


# ===========================================================================
# TestRelation
# ===========================================================================


class TestRelationConstruction:
    """Relation constructs and stores attributes correctly."""

    def test_stores_all_attributes_without_parameter(self) -> None:
        """Relation without parameter stores operator, arity, directed, and None parameter."""
        r = Relation("==", 2, False)
        assert r.operator == "=="
        assert r.arity == 2
        assert r.directed is False
        assert r.parameter is None

    def test_stores_parameter_when_provided(self) -> None:
        """Relation stores the parameter when one is supplied."""
        r = Relation("val>", 1, True, parameter=5)
        assert r.parameter == 5, f"Expected parameter 5, got {r.parameter}"


class TestRelationEquality:
    """Relation.__eq__ compares only operator and parameter."""

    def test_equal_when_operator_and_parameter_match(self) -> None:
        """Two relations with identical operator and parameter are equal."""
        r1 = Relation("==", 2, False)
        r2 = Relation("==", 2, True)  # arity and directed differ — must not matter
        assert r1 == r2, "Relations with same operator and parameter must be equal"

    def test_not_equal_when_operators_differ(self) -> None:
        """Relations with different operators are not equal."""
        r1 = Relation("==", 2, False)
        r2 = Relation("!=", 2, False)
        assert r1 != r2, "Relations with different operators must not be equal"

    def test_not_equal_when_parameters_differ(self) -> None:
        """Relations with different parameters are not equal."""
        r1 = Relation("val>", 1, True, parameter=5)
        r2 = Relation("val>", 1, True, parameter=10)
        assert r1 != r2, "Relations with different parameters must not be equal"

    def test_equal_relations_have_equal_hashes(self) -> None:
        """Equal relations must have equal hashes (hash contract)."""
        r1 = Relation("==", 2, False)
        r2 = Relation("==", 2, True)
        assert hash(r1) == hash(r2), "Equal relations must produce the same hash"

    def test_relation_usable_as_dict_key(self) -> None:
        """Relation can be used as a dictionary key without error."""
        r = Relation("==", 2, False)
        d = {r: "eq"}
        assert d[r] == "eq"


class TestRelationStr:
    """Relation.__str__ formats correctly with and without parameter."""

    def test_str_without_parameter(self) -> None:
        """__str__ omits params section when parameter is None."""
        r = Relation("==", 2, False)
        assert str(r) == "relation: ==", f"Unexpected __str__: {str(r)!r}"

    def test_str_with_parameter(self) -> None:
        """__str__ includes params section when parameter is set."""
        r = Relation("val>", 1, True, parameter=5)
        assert str(r) == "relation: val> | params: 5 ", f"Unexpected __str__: {str(r)!r}"


# ===========================================================================
# TestConstraint — concrete subclasses
# ===========================================================================


class TestConstraintConstruction:
    """Concrete constraint subclasses store scope and relation correctly."""

    def test_equal_stores_scope(self, var_x, var_y) -> None:
        """Equal stores the passed scope."""
        c = Equal([var_x, var_y])
        assert c.scope == [var_x, var_y]

    def test_equal_relation_operator(self, var_x, var_y) -> None:
        """Equal sets operator to '=='."""
        c = Equal([var_x, var_y])
        assert c.relation.operator == "=="

    def test_greater_relation_is_directed(self, var_x, var_y) -> None:
        """Greater sets directed=True on its relation."""
        c = Greater([var_x, var_y])
        assert c.relation.directed is True

    def test_equal_relation_is_undirected(self, var_x, var_y) -> None:
        """Equal sets directed=False on its relation."""
        c = Equal([var_x, var_y])
        assert c.relation.directed is False

    def test_equalval_stores_parameter(self, var_x) -> None:
        """EqualVal stores the parameter in its relation."""
        c = EqualVal([var_x], 5)
        assert c.relation.parameter == 5

    def test_distequal_stores_parameter(self, var_x, var_y) -> None:
        """DistEqual stores the parameter in its relation."""
        c = DistEqual([var_x, var_y], 3)
        assert c.relation.parameter == 3


class TestConstraintEquality:
    """Constraint.__eq__ respects directionality of the embedded relation."""

    def test_undirected_equal_regardless_of_scope_order(self, var_x, var_y) -> None:
        """Undirected constraints (Equal) with reversed scopes are equal."""
        c1 = Equal([var_x, var_y])
        c2 = Equal([var_y, var_x])
        assert c1 == c2, "Undirected Equal constraints must be order-independent"

    def test_directed_not_equal_when_scope_reversed(self, var_x, var_y) -> None:
        """Directed constraints (Greater) with reversed scopes are not equal."""
        c1 = Greater([var_x, var_y])
        c2 = Greater([var_y, var_x])
        assert c1 != c2, "Directed Greater constraints with reversed scopes must not be equal"

    def test_directed_equal_when_scope_and_type_match(self, var_x, var_y) -> None:
        """Directed constraints with identical scope order and type are equal."""
        c1 = Greater([var_x, var_y])
        c2 = Greater([var_x, var_y])
        assert c1 == c2

    def test_not_equal_when_constraint_types_differ(self, var_x, var_y) -> None:
        """Equal and NotEqual constraints on the same scope are not equal."""
        c1 = Equal([var_x, var_y])
        c2 = NotEqual([var_x, var_y])
        assert c1 != c2

    def test_not_equal_when_parameters_differ(self, var_x, var_y) -> None:
        """DistEqual constraints with different parameters are not equal."""
        c1 = DistEqual([var_x, var_y], 1)
        c2 = DistEqual([var_x, var_y], 2)
        assert c1 != c2


class TestConstraintHash:
    """Constraint.__hash__ is consistent with __eq__."""

    def test_undirected_equal_constraints_have_same_hash(self, var_x, var_y) -> None:
        """Undirected equal constraints produce the same hash."""
        c1 = Equal([var_x, var_y])
        c2 = Equal([var_y, var_x])
        assert hash(c1) == hash(c2), "Equal undirected constraints must share a hash"

    def test_directed_equal_constraints_have_same_hash(self, var_x, var_y) -> None:
        """Directed equal constraints produce the same hash."""
        c1 = Greater([var_x, var_y])
        c2 = Greater([var_x, var_y])
        assert hash(c1) == hash(c2)

    def test_constraints_usable_in_set(self, var_x, var_y) -> None:
        """Constraints can be stored in a set; equal undirected constraints deduplicate."""
        c1 = Equal([var_x, var_y])
        c2 = Equal([var_y, var_x])
        s = {c1, c2}
        assert len(s) == 1, "Equal undirected constraints must collapse to one set entry"

    def test_different_constraints_coexist_in_set(self, var_x, var_y) -> None:
        """Distinct constraints occupy separate set entries."""
        s = {Equal([var_x, var_y]), NotEqual([var_x, var_y]), Greater([var_x, var_y])}
        assert len(s) == 3, f"Expected 3 distinct entries, got {len(s)}"


class TestConstraintStr:
    """Constraint.__str__ produces the expected string."""

    def test_str_binary_constraint_contains_separator(self, var_x, var_y) -> None:
        """__str__ joins scope variable strings with ' && '."""
        c = Equal([var_x, var_y])
        result = str(c)
        assert "&&" in result, f"Expected '&&' separator in: {result!r}"

    def test_str_contains_relation(self, var_x, var_y) -> None:
        """__str__ includes the relation string."""
        c = Equal([var_x, var_y])
        result = str(c)
        assert "==" in result, f"Expected '==' in: {result!r}"


# ===========================================================================
# TestTargetNetwork
# ===========================================================================


class TestTargetNetworkConstruction:
    """TargetNetwork constructs correctly in both modes."""

    def test_default_mode_with_constraints(self, var_x, var_y) -> None:
        """Default mode constructs without error when constraints are provided."""
        tn = TargetNetwork(constraints=[Equal([var_x, var_y])], mode="default")
        assert tn.ask_counter == 0

    def test_default_mode_without_constraints_raises_value_error(self) -> None:
        """Default mode raises ValueError when no constraints are provided."""
        with pytest.raises(ValueError, match="Please provide constraints"):
            TargetNetwork(constraints=None, mode="default")

    def test_human_mode_does_not_require_constraints(self) -> None:
        """Human mode constructs without error even when constraints is None."""
        tn = TargetNetwork(constraints=None, mode="human")
        assert tn.mode == "human"

    def test_ask_counter_starts_at_zero(self, var_x, var_y) -> None:
        """ask_counter is initialised to 0 on construction."""
        tn = TargetNetwork(constraints=[Equal([var_x, var_y])])
        assert tn.ask_counter == 0, f"Expected 0, got {tn.ask_counter}"


class TestTargetNetworkAsk:
    """TargetNetwork.ask increments the counter and delegates to constraint.check."""

    def test_ask_increments_counter(self, var_x, var_y) -> None:
        """ask increments ask_counter by 1 on each call."""
        tn = TargetNetwork(constraints=[Equal([var_x, var_y])])
        tn.ask({"x": 1, "y": 1})
        tn.ask({"x": 2, "y": 3})
        assert tn.ask_counter == 2, f"Expected 2, got {tn.ask_counter}"

    def test_ask_returns_bool(self, var_x, var_y) -> None:
        """ask returns a boolean value."""
        tn = TargetNetwork(constraints=[Equal([var_x, var_y])])
        result = tn.ask({"x": 5, "y": 5})
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    def test_empty_constraint_list_always_returns_true(self, var_x) -> None:
        """ask returns True when the constraint list is empty (vacuous truth)."""
        tn = TargetNetwork(constraints=[])
        assert tn.ask({"x": 0}) is True


class TestTargetNetworkAskHuman:
    """TargetNetwork.ask in human mode reads from stdin."""

    def test_returns_true_on_yes_input(self, monkeypatch, var_x) -> None:
        """ask returns True when user types 'yes'."""
        monkeypatch.setattr("builtins.input", lambda: "yes")
        tn = TargetNetwork(mode="human")
        assert tn.ask({"x": 5}) is True

    def test_returns_false_on_no_input(self, monkeypatch, var_x) -> None:
        """ask returns False when user types 'no'."""
        monkeypatch.setattr("builtins.input", lambda: "no")
        tn = TargetNetwork(mode="human")
        assert tn.ask({"x": 5}) is False

    def test_returns_false_on_empty_input(self, monkeypatch) -> None:
        """ask returns False when user provides empty input."""
        monkeypatch.setattr("builtins.input", lambda: "")
        tn = TargetNetwork(mode="human")
        assert tn.ask({"x": 5}) is False

    def test_case_insensitive_yes(self, monkeypatch) -> None:
        """ask treats 'YES', 'Yes', etc. as True (strip+lower comparison)."""
        monkeypatch.setattr("builtins.input", lambda: "  YES  ")
        tn = TargetNetwork(mode="human")
        assert tn.ask({"x": 5}) is True

    def test_ask_increments_counter_in_human_mode(self, monkeypatch) -> None:
        """ask increments ask_counter in human mode just as in default mode."""
        monkeypatch.setattr("builtins.input", lambda: "yes")
        tn = TargetNetwork(mode="human")
        tn.ask({"x": 1})
        tn.ask({"x": 2})
        assert tn.ask_counter == 2, f"Expected 2, got {tn.ask_counter}"


class TestTargetNetworkStr:
    """TargetNetwork.__str__ reports constraint count and ask count."""

    def test_str_contains_constraint_and_ask_counts(self, var_x, var_y) -> None:
        """__str__ includes both the total constraint count and total ask count."""
        tn = TargetNetwork(constraints=[Equal([var_x, var_y])])
        tn.ask({"x": 5, "y": 5})
        result = str(tn)
        assert "1" in result, f"Expected counts in: {result!r}"
