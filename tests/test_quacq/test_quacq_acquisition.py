import pytest

from acquisition.baselines.quacq import GenerateExample
from core.base import Variable
from core.constraints import DistGreater, DistGreaterEqual, DistLess, DistLessEqual, DistNotEqual, EqualVal


# ===========================================================================
# Shared fixtures
# ===========================================================================


@pytest.fixture
def x() -> Variable:
    return Variable("x", [0, 10])


@pytest.fixture
def y() -> Variable:
    return Variable("y", [0, 10])


@pytest.fixture
def z() -> Variable:
    return Variable("z", [0, 10])


# ===========================================================================
# TestGenerateExample
# ===========================================================================


class TestGenerateExample:

    def test_returns_dict_with_all_variable_names(self, x, y) -> None:
        """GenerateExample dict keys must match the names of all vars."""
        c = EqualVal([x], 5)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None, "Should find an example violating x==5"
        assert set(example.keys()) == {"x", "y"}, (
            f"Expected keys {{'x', 'y'}}, got {set(example.keys())}"
        )

    def test_example_violates_at_least_one_B_constraint(self, x, y) -> None:
        """The returned example must violate at least one constraint in B."""
        c1 = EqualVal([x], 5)
        c2 = EqualVal([y], 7)
        B = {c1, c2}
        example = GenerateExample(B, [], [x, y])
        assert example is not None
        satisfied = [c.check(example) for c in B]
        assert not all(satisfied), (
            "GenerateExample must return an example that violates at least one B constraint"
        )

    def test_example_satisfies_all_L_constraints(self, x, y) -> None:
        """The example must respect every constraint in L."""
        c_pin_y = EqualVal([y], 3)
        c_in_B = EqualVal([x], 0)
        L = [{c_pin_y}]
        B = {c_in_B}
        example = GenerateExample(B, L, [x, y])
        assert example is not None
        assert example["y"] == 3, (
            f"L pins y=3 but example has y={example['y']}"
        )

    def test_empty_B_returns_none(self, x, y) -> None:
        """With B empty the model forces sum(bools)==0 != 0, which is UNSAT; must return None."""
        example = GenerateExample(set(), [], [x, y])
        assert example is None, "Empty B means no constraint to violate; must return None"

    def test_B_already_entailed_by_L_returns_none(self, x) -> None:
        """When L already forces all of B to be satisfied, GenerateExample must return None."""
        c = EqualVal([x], 5)
        L = [{c}]
        B = {c}
        example = GenerateExample(B, L, [x])
        assert example is None, (
            "L already forces B to be satisfied, so no violating example should exist"
        )

    def test_B_not_mutated(self, x, y) -> None:
        """GenerateExample must not remove or add constraints to B."""
        c = EqualVal([x], 5)
        B = {c}
        original_size = len(B)
        GenerateExample(B, [], [x, y])
        assert len(B) == original_size, "GenerateExample must not mutate B"

    def test_example_values_within_domain(self, x, y) -> None:
        """All values in the returned example must be within their variable domains."""
        c = EqualVal([x], 5)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None
        for var in [x, y]:
            val = example[var.name]
            assert var.domain[0] <= val <= var.domain[1], (
                f"{var.name}={val} is outside domain {var.domain}"
            )


# ===========================================================================
# TestGenerateExampleDistanceConstraint
# ===========================================================================


_DIST_CLASSES = [
    pytest.param(DistGreater,      id="DistGreater"),
    pytest.param(DistLess,         id="DistLess"),
    pytest.param(DistNotEqual,     id="DistNotEqual"),
    pytest.param(DistLessEqual,    id="DistLessEqual"),
    pytest.param(DistGreaterEqual, id="DistGreaterEqual"),
]

_PARAM = 3


class TestGenerateExampleDistanceConstraint:

    @pytest.mark.parametrize("cls", _DIST_CLASSES)
    def test_returns_non_none_example(self, x, y, cls) -> None:
        """GenerateExample must find a violating example for every distance constraint."""
        c = cls([x, y], _PARAM)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None, (
            f"GenerateExample returned None for {cls.__name__} with param={_PARAM}"
        )

    @pytest.mark.parametrize("cls", _DIST_CLASSES)
    def test_example_violates_distance_constraint(self, x, y, cls) -> None:
        """The returned example must violate the distance constraint in B."""
        c = cls([x, y], _PARAM)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None
        assert not c.check(example), (
            f"Example {example} satisfies {cls.__name__}(param={_PARAM}) but should violate it"
        )

    @pytest.mark.parametrize("cls", _DIST_CLASSES)
    def test_example_values_within_domain(self, x, y, cls) -> None:
        """All values in the returned example must lie within their variable domains."""
        c = cls([x, y], _PARAM)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None
        for var in [x, y]:
            val = example[var.name]
            assert var.domain[0] <= val <= var.domain[1], (
                f"{var.name}={val} outside domain {var.domain} for {cls.__name__}"
            )

    @pytest.mark.parametrize("cls", _DIST_CLASSES)
    def test_example_contains_all_variable_keys(self, x, y, cls) -> None:
        """Returned example dict must contain every variable passed to GenerateExample."""
        c = cls([x, y], _PARAM)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None
        assert set(example.keys()) == {"x", "y"}, (
            f"Expected keys {{'x', 'y'}}, got {set(example.keys())}"
        )

    def test_distance_gt_entailed_by_L_returns_none(self, x, y) -> None:
        """When L already forces DistGreater, no violating example exists; must return None."""
        c = DistGreater([x, y], _PARAM)
        example = GenerateExample({c}, [{c}], [x, y])
        assert example is None, (
            "L forces |x-y|>3 so B cannot be violated; expected None"
        )

    def test_distance_lt_entailed_by_L_returns_none(self, x, y) -> None:
        """When L already forces DistLess, no violating example exists; must return None."""
        c = DistLess([x, y], _PARAM)
        example = GenerateExample({c}, [{c}], [x, y])
        assert example is None, (
            "L forces |x-y|<3 so B cannot be violated; expected None"
        )

    def test_distance_constraint_respects_L_pinned_variable(self, x, y, z) -> None:
        """GenerateExample must satisfy L (pinned z=2) while violating DistLess on x, y."""
        c_L = EqualVal([z], 2)
        c_B = DistLess([x, y], _PARAM)
        example = GenerateExample({c_B}, [{c_L}], [x, y, z])
        assert example is not None
        assert example["z"] == 2, (
            f"L pins z=2 but example has z={example['z']}"
        )
        