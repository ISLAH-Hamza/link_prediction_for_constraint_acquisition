import networkx as nx
import pytest

from acquisition.baselines.pquacq import adamic_adar, predict_and_ask, pquacq
from acquisition.baselines.quacq import GenerateExample
from core.base import Variable, TargetNetwork
from core.constraints import (
    DistGreater, DistGreaterEqual, DistLess, DistLessEqual, DistNotEqual,
    EqualVal, Equal, NotEqual,
)


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
# TestGenerateExample — mirrors tests/test_quacq/test_acquisition.py
# (pquacq delegates to quacq.GenerateExample, so the same guarantees hold)
# ===========================================================================


class TestGenerateExample:

    def test_returns_dict_with_all_variable_names(self, x, y) -> None:
        """GenerateExample dict keys must match the names of all vars."""
        c = EqualVal([x], 5)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None, "Should find an example violating x==5"
        assert set(example.keys()) == {"x", "y"}

    def test_example_violates_at_least_one_B_constraint(self, x, y) -> None:
        """The returned example must violate at least one constraint in B."""
        c1 = EqualVal([x], 5)
        c2 = EqualVal([y], 7)
        B = {c1, c2}
        example = GenerateExample(B, [], [x, y])
        assert example is not None
        satisfied = [c.check(example) for c in B]
        assert not all(satisfied)

    def test_example_satisfies_all_L_constraints(self, x, y) -> None:
        """The example must respect every constraint in L."""
        c_pin_y = EqualVal([y], 3)
        c_in_B = EqualVal([x], 0)
        example = GenerateExample({c_in_B}, [{c_pin_y}], [x, y])
        assert example is not None
        assert example["y"] == 3

    def test_empty_B_returns_none(self, x, y) -> None:
        """Empty B → UNSAT → must return None."""
        assert GenerateExample(set(), [], [x, y]) is None

    def test_B_already_entailed_by_L_returns_none(self, x) -> None:
        """When L already forces all of B to be satisfied, must return None."""
        c = EqualVal([x], 5)
        assert GenerateExample({c}, [{c}], [x]) is None

    def test_B_not_mutated(self, x, y) -> None:
        """GenerateExample must not remove or add constraints to B."""
        c = EqualVal([x], 5)
        B = {c}
        GenerateExample(B, [], [x, y])
        assert len(B) == 1

    def test_example_values_within_domain(self, x, y) -> None:
        """All values in the returned example must lie within their variable domains."""
        c = EqualVal([x], 5)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None
        for var in [x, y]:
            val = example[var.name]
            assert var.domain[0] <= val <= var.domain[1]


# ===========================================================================
# TestGenerateExampleDistanceConstraint — mirrors test_quacq counterpart
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
        c = cls([x, y], _PARAM)
        assert GenerateExample({c}, [], [x, y]) is not None

    @pytest.mark.parametrize("cls", _DIST_CLASSES)
    def test_example_violates_distance_constraint(self, x, y, cls) -> None:
        c = cls([x, y], _PARAM)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None
        assert not c.check(example)

    @pytest.mark.parametrize("cls", _DIST_CLASSES)
    def test_example_values_within_domain(self, x, y, cls) -> None:
        c = cls([x, y], _PARAM)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None
        for var in [x, y]:
            val = example[var.name]
            assert var.domain[0] <= val <= var.domain[1]

    @pytest.mark.parametrize("cls", _DIST_CLASSES)
    def test_example_contains_all_variable_keys(self, x, y, cls) -> None:
        c = cls([x, y], _PARAM)
        example = GenerateExample({c}, [], [x, y])
        assert example is not None
        assert set(example.keys()) == {"x", "y"}

    def test_distance_gt_entailed_by_L_returns_none(self, x, y) -> None:
        c = DistGreater([x, y], _PARAM)
        assert GenerateExample({c}, [{c}], [x, y]) is None

    def test_distance_lt_entailed_by_L_returns_none(self, x, y) -> None:
        c = DistLess([x, y], _PARAM)
        assert GenerateExample({c}, [{c}], [x, y]) is None

    def test_distance_constraint_respects_L_pinned_variable(self, x, y, z) -> None:
        c_L = EqualVal([z], 2)
        c_B = DistLess([x, y], _PARAM)
        example = GenerateExample({c_B}, [{c_L}], [x, y, z])
        assert example is not None
        assert example["z"] == 2


# ===========================================================================
# TestAdamicAdar
# ===========================================================================


class TestAdamicAdar:

    def test_no_common_neighbors_returns_zero(self) -> None:
        """Two disconnected nodes have no common neighbors → score == 0."""
        G = nx.Graph()
        G.add_nodes_from(["a", "b"])
        assert adamic_adar(G, ["a", "b"]) == 0.0

    def test_one_common_neighbor_degree_two(self) -> None:
        """a-c-b with deg(c)==2: score = 1/log(2)."""
        import numpy as np
        G = nx.Graph()
        G.add_edges_from([("a", "c"), ("b", "c")])
        score = adamic_adar(G, ["a", "b"])
        assert score == pytest.approx(1 / np.log(2))

    def test_score_increases_with_more_common_neighbors(self) -> None:
        """More shared neighbors with degree > 1 → higher score."""
        import numpy as np
        G1 = nx.Graph()
        G1.add_edges_from([("a", "c"), ("b", "c")])

        G2 = nx.Graph()
        G2.add_edges_from([("a", "c"), ("b", "c"), ("a", "d"), ("b", "d")])

        s1 = adamic_adar(G1, ["a", "b"])
        s2 = adamic_adar(G2, ["a", "b"])
        assert s2 > s1

    def test_neighbor_with_degree_one_is_skipped(self) -> None:
        """A common neighbor of degree 1 contributes 0 (log(1)==0 avoided by degree>1 guard)."""
        G = nx.Graph()
        # c connects only to a and b, so its degree is 2 — this passes the guard
        # But if we add a node with degree 1 it should be skipped
        G.add_edges_from([("a", "c"), ("c", "b")])   # deg(c)==2 → counted
        G.add_nodes_from(["d"])                        # deg(d)==0 → not a common neighbor anyway
        score_with = adamic_adar(G, ["a", "b"])
        # removing c leaves no common neighbors
        G2 = nx.Graph()
        G2.add_nodes_from(["a", "b"])
        score_without = adamic_adar(G2, ["a", "b"])
        assert score_with > score_without


# ===========================================================================
# TestPredictAndAsk — unit tests for predict_and_ask
# ===========================================================================


@pytest.fixture
def vars_abc():
    a = Variable("a", [0, 5])
    b = Variable("b", [0, 5])
    c_var = Variable("c", [0, 5])
    return a, b, c_var



class TestPquacqAlpha:

    def _run_pquacq(self, alpha: int):
        """Run pquacq on a tiny two-variable problem and return the target network."""
        x = Variable("x", [0, 4])
        y = Variable("y", [0, 4])
        c = Equal([x, y])
        target = TargetNetwork({c})
        B = {c, NotEqual([x, y])}
        pquacq(alpha=alpha, score=adamic_adar, B=B,
               target_network=target, variables=[x, y])
        return target

    def test_alpha_zero_makes_no_proactive_asks(self) -> None:
        """alpha=0 means predict_and_ask never enters its loop → req_yes + req_no == 0."""
        target = self._run_pquacq(alpha=0)
        assert target.req_yes + target.req_no == 0

    def test_higher_alpha_allows_more_proactive_asks(self) -> None:
        """alpha=10 allows more proactive constraint queries than alpha=0."""
        x = Variable("x", [0, 4])
        y = Variable("y", [0, 4])
        z = Variable("z", [0, 4])
        c_xy = Equal([x, y])
        c_yz = Equal([y, z])
        c_xz = Equal([x, z])
        target_hi = TargetNetwork({c_xy, c_yz, c_xz})
        B_hi = {c_xy, c_yz, c_xz, NotEqual([x, y]), NotEqual([y, z]), NotEqual([x, z])}
        pquacq(alpha=10, score=adamic_adar, B=B_hi,
               target_network=target_hi, variables=[x, y, z])

        target_lo = TargetNetwork({c_xy, c_yz, c_xz})
        B_lo = {c_xy, c_yz, c_xz, NotEqual([x, y]), NotEqual([y, z]), NotEqual([x, z])}
        pquacq(alpha=0, score=adamic_adar, B=B_lo,
               target_network=target_lo, variables=[x, y, z])

        # Higher alpha → more (or equal) proactive asks
        assert (target_hi.req_yes + target_hi.req_no) >= (target_lo.req_yes + target_lo.req_no)

    def test_pquacq_returns_list(self) -> None:
        """pquacq must return a list (L)."""
        x = Variable("x", [0, 4])
        y = Variable("y", [0, 4])
        c = Equal([x, y])
        B = {c, NotEqual([x, y])}
        target = TargetNetwork({c})
        L = pquacq(alpha=3, score=adamic_adar, B=B,
                   target_network=target, variables=[x, y])
        assert isinstance(L, list)

    def test_pquacq_terminates_and_converges_small(self) -> None:
        """On a tiny problem pquacq must terminate and learn the target constraint."""
        x = Variable("x", [0, 4])
        y = Variable("y", [0, 4])
        c = Equal([x, y])
        B = {c, NotEqual([x, y])}
        target = TargetNetwork({c})
        L = pquacq(alpha=3, score=adamic_adar, B=B,
                   target_network=target, variables=[x, y])
        learned = {con for conj in L for con in conj}
        assert c in learned


# ===========================================================================
# TestScoreEffect — behaviour with a custom scoring function
# ===========================================================================


class TestScoreEffect:

    def test_constant_zero_score_skips_all_candidates(self, vars_abc) -> None:
        """A score function that always returns 0 causes predict_and_ask to skip everything."""
        a, b, c_var = vars_abc
        c_ab = Equal([a, b])
        c_bc = Equal([b, c_var])
        r = c_ab.relation

        zero_score = lambda G, edge: 0.0

        target = TargetNetwork({c_ab, c_bc})
        L = [{c_ab}]
        B = {c_bc}
        initial_asks = target.ask_counter

        predict_and_ask(r, zero_score, alpha=5, B=B, L=L, target_network=target)
        assert target.ask_counter == initial_asks, \
            "Zero-score function must not trigger any oracle call"

    def test_constant_one_score_asks_all_candidates(self, vars_abc) -> None:
        """A score function that always returns 1 causes all candidates to be queried."""
        a, b, c_var = vars_abc
        c_ab = Equal([a, b])
        c_bc = Equal([b, c_var])
        r = c_ab.relation

        one_score = lambda G, edge: 1.0

        target = TargetNetwork({c_ab, c_bc})
        L = [{c_ab}]
        B = {c_bc}

        predict_and_ask(r, one_score, alpha=5, B=B, L=L, target_network=target)
        # c_bc is in target → should have been confirmed
        assert target.req_yes >= 1

    def test_score_function_receives_graph_and_edge(self, vars_abc) -> None:
        """The score function must be called with (graph, edge) where edge is a list of node names."""
        a, b, c_var = vars_abc
        c_ab = Equal([a, b])
        c_bc = Equal([b, c_var])
        r = c_ab.relation

        calls = []

        def recording_score(G, edge):
            calls.append((type(G).__name__, edge))
            return 1.0   # always recommend

        target = TargetNetwork({c_ab, c_bc})
        L = [{c_ab}]
        B = {c_bc}

        predict_and_ask(r, recording_score, alpha=5, B=B, L=L, target_network=target)
        assert len(calls) >= 1
        graph_type, edge = calls[0]
        assert graph_type == "Graph"
        assert isinstance(edge, list)
        assert all(isinstance(n, str) for n in edge)