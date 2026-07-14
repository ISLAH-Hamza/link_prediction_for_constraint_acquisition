import pytest
import numpy as np
import networkx as nx
from unittest.mock import MagicMock

from core.base import Variable
from core.constraints import Equal, NotEqual
from acquisition.ml_quacq.ml_quacq_e import (
    compute_features,
    graph_drop_edges,
    fit_link_predictor,
    predict_and_ask,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def vars5():
    return [Variable(str(i), [0, 10]) for i in range(5)]


@pytest.fixture
def undirected_graph():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)])
    return G


@pytest.fixture
def directed_graph():
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])
    return G


# Precomputed infomap modules to avoid running infomap in unit tests
MOCK_MODULES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2}


# ===========================================================================
# TestFeatures
# ===========================================================================

class TestFeatures:

    def test_output_shape(self, undirected_graph):
        vec = compute_features(undirected_graph, (0, 1), infomap_modules=MOCK_MODULES)
        assert vec.shape == (7,)

    def test_returns_zeros_for_missing_node(self, undirected_graph):
        vec = compute_features(undirected_graph, (0, 99), infomap_modules=MOCK_MODULES)
        np.testing.assert_array_equal(vec, np.zeros(7))

    def test_undirected_degree_slots(self, undirected_graph):
        # slots: [jaccard, ra, infomap, deg(u), deg(u), deg(v), deg(v)]
        vec = compute_features(undirected_graph, (0, 1), infomap_modules=MOCK_MODULES)
        assert vec[3] == vec[4] == undirected_graph.degree(0)
        assert vec[5] == vec[6] == undirected_graph.degree(1)


    def test_jaccard_is_in_unit_interval(self, undirected_graph):
        vec = compute_features(undirected_graph, (0, 1), infomap_modules=MOCK_MODULES)
        assert 0.0 <= vec[0] <= 1.0

    def test_resource_allocation_nonnegative(self, undirected_graph):
        vec = compute_features(undirected_graph, (0, 1), infomap_modules=MOCK_MODULES)
        assert vec[1] >= 0.0

    def test_infomap_same_community(self, undirected_graph):
        # nodes 0 and 1 are in community 1
        vec = compute_features(undirected_graph, (0, 1), infomap_modules=MOCK_MODULES)
        assert vec[2] == 1.0

    def test_infomap_different_community(self, undirected_graph):
        # node 0 is in community 1, node 2 is in community 2
        vec = compute_features(undirected_graph, (0, 2), infomap_modules=MOCK_MODULES)
        assert vec[2] == 0.0

    def test_isolated_nodes_have_zero_degrees(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        vec = compute_features(G, (0, 1), infomap_modules={0: 1, 1: 1})
        assert vec[3] == vec[4] == vec[5] == vec[6] == 0


# ===========================================================================
# TestGraphDropEdges
# ===========================================================================

class TestGraphDropEdges:

    def test_original_not_mutated(self, undirected_graph):
        original_edges = set(undirected_graph.edges())
        graph_drop_edges(undirected_graph, n=2)
        assert set(undirected_graph.edges()) == original_edges

    def test_result_has_fewer_edges(self, undirected_graph):
        result = graph_drop_edges(undirected_graph, n=2)
        assert result.number_of_edges() == undirected_graph.number_of_edges() - 2

    def test_default_drops_one_edge(self, undirected_graph):
        result = graph_drop_edges(undirected_graph)
        assert result.number_of_edges() == undirected_graph.number_of_edges() - 1

    def test_nodes_are_preserved(self, undirected_graph):
        result = graph_drop_edges(undirected_graph, n=3)
        assert set(result.nodes()) == set(undirected_graph.nodes())

    def test_n_larger_than_edge_count_empties_graph(self, undirected_graph):
        result = graph_drop_edges(undirected_graph, n=undirected_graph.number_of_edges() + 10)
        assert result.number_of_edges() == 0


# ===========================================================================
# TestFitLinkPredictor
# ===========================================================================

_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # 10 unique pairs


class TestFitLinkPredictor:

    @pytest.fixture
    def graph_with_edges(self):
        G = nx.Graph()
        G.add_nodes_from(range(5))
        for i, j in _PAIRS[:5]:
            G.add_edge(i, j)
        return G

    @pytest.fixture
    def positive_constraints(self, vars5):
        return [Equal([vars5[i], vars5[j]]) for i, j in _PAIRS]

    @pytest.fixture
    def negative_constraints(self, vars5):
        return [NotEqual([vars5[i], vars5[j]]) for i, j in _PAIRS]



    def test_returns_model_with_sufficient_data(
        self, graph_with_edges, positive_constraints, negative_constraints
    ):
        model, accuracy = fit_link_predictor(
            graph_with_edges, positive_constraints, negative_constraints
        )
        assert model is not None
        assert 0.0 <= accuracy <= 1.0

    def test_predict_proba_output_shape(
        self, graph_with_edges, positive_constraints, negative_constraints
    ):
        model, _ = fit_link_predictor(
            graph_with_edges, positive_constraints, negative_constraints
        )
        if model is not None:
            proba = model.predict_proba([0, 1])
            assert proba.shape == (1, 2)

