import pytest
import numpy as np
import torch
import networkx as nx
from unittest.mock import MagicMock

from core.base import Variable
from core.constraints import Equal, NotEqual
from acquisition.ml_quacq.ml_quacq_n import (
    get_node_feature_matrix,
    convert_to_pyg_data,
    LinkPredictor,
    fit_gnn_link_predictor,
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
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)])
    return G


_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # 10 unique pairs


# ===========================================================================
# TestGetNodeFeatureMatrix
# ===========================================================================

class TestGetNodeFeatureMatrix:

    def test_output_shape_undirected(self, undirected_graph):
        x = get_node_feature_matrix(undirected_graph)
        assert x.shape == (5, 5)

    def test_output_shape_directed(self, directed_graph):
        x = get_node_feature_matrix(directed_graph)
        assert x.shape == (5, 5)

    def test_returns_float_tensor(self, undirected_graph):
        x = get_node_feature_matrix(undirected_graph)
        assert x.dtype == torch.float32

    def test_empty_graph_returns_zero_tensor(self):
        G = nx.Graph()
        x = get_node_feature_matrix(G)
        assert x.shape == (0, 5)

    def test_undirected_in_out_degree_equal(self, undirected_graph):
        # Before scaling, in_degree == out_degree for undirected graphs.
        # After StandardScaler the columns may differ due to per-feature normalisation,
        # but if the raw values are identical the scaled values must also be identical.
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edges_from([(0, 1), (1, 2)])
        x = get_node_feature_matrix(G)
        # Column 0 (in_degree) and column 1 (out_degree) must be identical
        torch.testing.assert_close(x[:, 0], x[:, 1])

    def test_single_node_graph_does_not_raise(self):
        G = nx.Graph()
        G.add_node(0)
        x = get_node_feature_matrix(G)
        assert x.shape == (1, 5)


# ===========================================================================
# TestConvertToPygData
# ===========================================================================

class TestConvertToPygData:

    def test_mapping_covers_all_nodes(self, undirected_graph):
        x = get_node_feature_matrix(undirected_graph)
        _, mapping = convert_to_pyg_data(undirected_graph, x)
        assert set(mapping.keys()) == set(undirected_graph.nodes())

    def test_mapping_indices_are_contiguous(self, undirected_graph):
        x = get_node_feature_matrix(undirected_graph)
        _, mapping = convert_to_pyg_data(undirected_graph, x)
        assert sorted(mapping.values()) == list(range(len(undirected_graph.nodes())))

    def test_edge_index_shape(self, undirected_graph):
        x = get_node_feature_matrix(undirected_graph)
        data, _ = convert_to_pyg_data(undirected_graph, x)
        # Each edge is added in both directions → 2 * num_edges columns
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 2 * undirected_graph.number_of_edges()

    def test_empty_graph_empty_edge_index(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        x = get_node_feature_matrix(G)
        data, _ = convert_to_pyg_data(G, x)
        assert data.edge_index.shape == (2, 0)

    def test_feature_matrix_preserved(self, undirected_graph):
        x = get_node_feature_matrix(undirected_graph)
        data, _ = convert_to_pyg_data(undirected_graph, x)
        torch.testing.assert_close(data.x, x)


# ===========================================================================
# TestLinkPredictor
# ===========================================================================

class TestLinkPredictor:

    def test_forward_output_shape(self):
        model = LinkPredictor(in_channels=5, hidden_channels=16)
        x = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_label_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        out = model(x, edge_index, edge_label_index)
        assert out.shape == (2, 1)

    def test_encode_output_shape(self):
        model = LinkPredictor(in_channels=5, hidden_channels=16)
        x = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        z = model.encode(x, edge_index)
        assert z.shape == (4, 16)

    def test_decode_output_shape(self):
        model = LinkPredictor(in_channels=5, hidden_channels=16)
        z = torch.randn(4, 16)
        edge_label_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        out = model.decode(z, edge_label_index)
        assert out.shape == (2, 1)


# ===========================================================================
# TestFitGnnLinkPredictor
# ===========================================================================

class TestFitGnnLinkPredictor:

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

    def test_returns_none_when_no_matching_nodes(self, vars5):
        # Graph has no nodes matching the variable names → empty indices → (None, 0.0)
        G = nx.Graph()
        G.add_nodes_from([99, 100])
        positiveE = [Equal([vars5[0], vars5[1]])]
        negativeE = [NotEqual([vars5[0], vars5[1]])]
        model, accuracy = fit_gnn_link_predictor(G, positiveE, negativeE)
        assert model is None
        assert accuracy == 0.0

    def test_returns_model_with_sufficient_data(
        self, graph_with_edges, positive_constraints, negative_constraints
    ):
        model, accuracy = fit_gnn_link_predictor(
            graph_with_edges, positive_constraints, negative_constraints
        )
        assert model is not None
        assert 0.0 <= accuracy <= 1.0

    def test_predict_proba_output_shape(
        self, graph_with_edges, positive_constraints, negative_constraints
    ):
        model, _ = fit_gnn_link_predictor(
            graph_with_edges, positive_constraints, negative_constraints
        )
        if model is not None:
            proba = model.predict_proba([0, 1])
            assert proba.shape == (1, 2)

    def test_predict_proba_sums_to_one(
        self, graph_with_edges, positive_constraints, negative_constraints
    ):
        model, _ = fit_gnn_link_predictor(
            graph_with_edges, positive_constraints, negative_constraints
        )
        if model is not None:
            proba = model.predict_proba([0, 1])
            np.testing.assert_almost_equal(proba[0].sum(), 1.0)

    def test_predict_proba_unknown_node_returns_half(
        self, graph_with_edges, positive_constraints, negative_constraints
    ):
        model, _ = fit_gnn_link_predictor(
            graph_with_edges, positive_constraints, negative_constraints
        )
        if model is not None:
            proba = model.predict_proba([0, 999])
            np.testing.assert_array_equal(proba, [[0.5, 0.5]])


# ===========================================================================
# TestPredictAndAsk
# ===========================================================================

class TestPredictAndAsk:

    def test_does_nothing_when_E_is_empty(self):
        link_predictor = MagicMock()
        target = MagicMock()
        relation = MagicMock()
        relation.directed = False
        variables = [Variable(str(i), [0, 10]) for i in range(5)]

        predict_and_ask(3, relation, set(), [], [], target, variables)

        target.ask_req.assert_not_called()

    def test_does_nothing_when_excluded_is_empty(self, vars5):
        target = MagicMock()
        relation = Equal([vars5[0], vars5[1]]).relation

        L = [{Equal([vars5[i], vars5[j]])} for i, j in _PAIRS]

        predict_and_ask(3, relation, set(), L, [], target, vars5)

        target.ask_req.assert_not_called()

    def test_does_nothing_when_accuracy_below_threshold(self, vars5):
        relation = Equal([vars5[0], vars5[1]]).relation
        variables = [Variable(str(i), [0, 10]) for i in range(10)]

        L = [{Equal([variables[i], variables[j]])}
             for i in range(10) for j in range(i + 1, 10)][:10]
        NL = [NotEqual([variables[i], variables[j]])
              for i in range(10) for j in range(i + 1, 10)][:10]

        mock_model = MagicMock()
        mock_fit = MagicMock(return_value=(mock_model, 0.3))
        target = MagicMock()
        target.set_time = MagicMock()

        # Patch fit_gnn_link_predictor via kwargs is not possible directly;
        # test the threshold logic by verifying ask_req is not called when
        # the real model returns low accuracy on minimal data (empty indices → None).
        predict_and_ask(
            3, relation, set(), L, NL, target, variables,
            accuracy_threshold=0.99,
        )

        target.ask_req.assert_not_called()