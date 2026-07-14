import networkx as nx
import numpy as np
import random, time, tqdm
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.data import Data


# local imports
from core.base import TargetNetwork
from acquisition.baselines import quacq
from core.solve import equivalent
from utils import writelog


def graph_drop_edges(G, n=1, seed=None):
    """
    Returns a copy of G with n randomly removed edges.
    """
    graph = G.copy()
    edges = list(graph.edges())

    rng = random.Random(seed)
    for _ in range(n):
        if not edges:
            break
        edge = rng.choice(edges)
        graph.remove_edge(*edge)
        edges.remove(edge)

    return graph




USER_FEATURES = ("degree", "betweenness", "closeness")
ALL_FEATURES = ("in_degree", "out_degree", "betweenness", "in_closeness", "out_closeness")

# User-facing names expand to internal feature columns.
# "degree" -> both in/out degree; "closeness" -> both in/out closeness.
_FEATURE_EXPANSION = {
    "degree": ("in_degree", "out_degree"),
    "betweenness": ("betweenness",),
    "closeness": ("in_closeness", "out_closeness"),
}


def _expand_features(features):
    """
    Expand user-facing feature names to their internal columns.
    Accepts any mix of USER_FEATURES and ALL_FEATURES; preserves order, drops dups.
    """
    if features is None:
        features = USER_FEATURES

    expanded = []
    for f in features:
        if f in _FEATURE_EXPANSION:
            for sub in _FEATURE_EXPANSION[f]:
                if sub not in expanded:
                    expanded.append(sub)
        elif f in ALL_FEATURES:
            if f not in expanded:
                expanded.append(f)
        else:
            raise ValueError(
                f"Unknown feature: {f!r}. Available user features: {USER_FEATURES}; "
                f"or low-level: {ALL_FEATURES}"
            )
    return tuple(expanded)


def get_node_feature_matrix(G, features=None) -> torch.FloatTensor:
    """
    Returns Tensor (NumNodes, len(features)) with the requested node features.
    Rows correspond to nodes sorted by int(node).

    User-facing features: degree, betweenness, closeness.
      - "degree"    expands to in_degree + out_degree
      - "closeness" expands to in_closeness + out_closeness
    For undirected graphs: in_degree==out_degree and in_closeness==out_closeness.

    If `features` is None, all user features are used.
    """
    features = _expand_features(features)

    nodes = sorted(list(G.nodes()), key=lambda x: int(x))

    needs_betweenness  = "betweenness" in features
    needs_in_closeness = "in_closeness" in features
    needs_out_closeness = "out_closeness" in features

    betweenness   = nx.betweenness_centrality(G, normalized=True) if needs_betweenness else {}
    out_closeness = nx.closeness_centrality(G) if needs_out_closeness else {}
    if needs_in_closeness:
        in_closeness = (
            nx.closeness_centrality(G.reverse(copy=False))
            if isinstance(G, nx.DiGraph) else nx.closeness_centrality(G)
        )
    else:
        in_closeness = {}

    n = len(nodes)
    norm = float(n - 1) if n > 1 else 1.0

    def _value(feat, u):
        if feat == "in_degree":
            return float(G.in_degree(u)) / norm if G.is_directed() else float(G.degree(u)) / norm
        if feat == "out_degree":
            return float(G.out_degree(u)) / norm if G.is_directed() else float(G.degree(u)) / norm
        if feat == "betweenness":
            return float(betweenness.get(u, 0.0))
        if feat == "in_closeness":
            return float(in_closeness.get(u, 0.0))
        if feat == "out_closeness":
            return float(out_closeness.get(u, 0.0))
        return 0.0

    matrix = [[_value(f, u) for f in features] for u in nodes]
    data = np.array(matrix, dtype=float)

    if data.shape[0] == 0:
        return torch.zeros((0, len(features)), dtype=torch.float32)

    scaler = StandardScaler()
    try:
        data = scaler.fit_transform(data)
    except Exception:
        pass

    return torch.tensor(data, dtype=torch.float32)



# =============================================================================
# 1. THE GNN MODEL ARCHITECTURE
# =============================================================================

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        # --- ENCODER (GNN) ---
        # Stack `num_layers` GINConv layers; each one expands the receptive field
        # by one hop (layer k aggregates k-hop neighborhoods).
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                )
            ))

        # --- DECODER (Link Prediction Head) ---
        # We concatenate embeddings of node U and node V, so input is hidden*2
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1) # Outputs a single score

    def encode(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        return x # Returns Matrix Z (N_nodes x Hidden_dim)

    def decode(self, z, edge_label_index):
        # z: All node embeddings
        # edge_label_index: The specific pairs [u, v] we want to predict
        
        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]

        z_src = z[src_idx]
        z_dst = z[dst_idx]

        # Concatenate features of the two nodes
        z_edge = torch.cat([z_src, z_dst], dim=-1)

        # Classify
        out = self.lin1(z_edge)
        out = F.relu(out)
        out = self.lin2(out)
        return out

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

# =============================================================================
# 2. DATA CONVERSION HELPERS
# =============================================================================

def convert_to_pyg_data(G, feature_matrix):
    """
    Converts NetworkX graph and Feature Matrix to PyTorch Geometric Data object.
    Returns: Data, mapping (dict)
    """
    # 1. Create a Mapping: Node Name (e.g. "25") -> Internal Index (0..N)
    sorted_nodes = sorted(list(G.nodes()), key=lambda x: int(x))
    mapping = {node: i for i, node in enumerate(sorted_nodes)}

    # 2. Build Edge Index (directed graphs keep arc direction; undirected add both)
    edges = []
    for u, v in G.edges():
        if u in mapping and v in mapping:
            edges.append([mapping[u], mapping[v]])
            if not G.is_directed():
                edges.append([mapping[v], mapping[u]])

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=feature_matrix, edge_index=edge_index), mapping

# =============================================================================
# 3. TRAINING AND WRAPPER
# =============================================================================


def fit_gnn_link_predictor(G, positiveE, negativeE, seed=42, features=None, num_layers=2, hidden_channels=32, **kwargs):
    """
    Trains the GNN on the current graph and returns a wrapper for prediction.

    Train/val split (80/20) with transductive link prediction:
      - Training: positive supervision edges are hidden from MP; negative edges
        are never in MP.  MP uses only the val-split positive edges.
      - Validation: MP uses all training positive edges.  Val positive edges
        (the ones being predicted) are hidden.
    """
    torch.manual_seed(seed)

    # 1. Prepare Data
    x = get_node_feature_matrix(G, features=features)
    pyg_data, mapping = convert_to_pyg_data(G, x)

    # 2. Prepare edge indices for positives and negatives
    pos_indices = []
    for c in positiveE:
        try:
            u_name = int(c.scope[0].name)
            v_name = int(c.scope[1].name)
            if u_name in mapping and v_name in mapping:
                pos_indices.append([mapping[u_name], mapping[v_name]])
        except: pass

    neg_indices = []
    for c in negativeE:
        try:
            u_name = int(c.scope[0].name)
            v_name = int(c.scope[1].name)
            if u_name in mapping and v_name in mapping:
                neg_indices.append([mapping[u_name], mapping[v_name]])
        except: pass

    if not pos_indices or not neg_indices:
        return None, 0.0

    pos_edge_index = torch.tensor(pos_indices, dtype=torch.long).t()
    neg_edge_index = torch.tensor(neg_indices, dtype=torch.long).t()

    # 3. 80/20 train/val split
    n_pos = pos_edge_index.size(1)
    n_neg = neg_edge_index.size(1)

    perm_pos = torch.randperm(n_pos)
    perm_neg = torch.randperm(n_neg)

    n_train_pos = int(0.8 * n_pos)
    n_train_neg = int(0.8 * n_neg)

    train_pos = pos_edge_index[:, perm_pos[:n_train_pos]]
    val_pos   = pos_edge_index[:, perm_pos[n_train_pos:]]

    train_neg = neg_edge_index[:, perm_neg[:n_train_neg]]
    val_neg   = neg_edge_index[:, perm_neg[n_train_neg:]]

    # 4. Build message-passing edge indices from G (directed or undirected per relation)
    mp_edge_index = pyg_data.edge_index
    train_mp_edge_index = mp_edge_index
    val_mp_edge_index = mp_edge_index

    # 5. Supervision labels
    train_edge_label_index = torch.cat([train_pos, train_neg], dim=1)
    train_labels = torch.cat([
        torch.ones(train_pos.size(1)),
        torch.zeros(train_neg.size(1))
    ])

    val_edge_label_index = torch.cat([val_pos, val_neg], dim=1)
    val_labels = torch.cat([
        torch.ones(val_pos.size(1)),
        torch.zeros(val_neg.size(1))
    ])

    # 6. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinkPredictor(in_channels=x.size(1), hidden_channels=hidden_channels, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    node_features = pyg_data.x.to(device)
    train_mp_edge_index = train_mp_edge_index.to(device)
    val_mp_edge_index = val_mp_edge_index.to(device)
    train_edge_label_index = train_edge_label_index.to(device)
    train_labels = train_labels.to(device)
    val_edge_label_index = val_edge_label_index.to(device)
    val_labels = val_labels.to(device)

    # 7. Train Loop
    model.train()

    # Data augmentation: when labeled arcs are scarce, also train on perturbed graphs
    augmented_data = []
    if len(positiveE) + len(negativeE) < 100:
        for i in range(10):
            G_aug = graph_drop_edges(G, seed=seed + i + len(positiveE))
            x_aug = get_node_feature_matrix(G_aug, features=features)
            pyg_aug, _ = convert_to_pyg_data(G_aug, x_aug)
            augmented_data.append((pyg_aug.x.to(device), pyg_aug.edge_index.to(device)))

    smote = SMOTE(random_state=seed)
    rng = random.Random(seed)
    smote_data = None  # cached SMOTE result, refreshed every 10 epochs
    y_np = train_labels.cpu().numpy()

    for epoch in range(100):
        # Standard forward pass: gradients flow through encoder + decoder
        optimizer.zero_grad()
        out = model(node_features, train_mp_edge_index, train_edge_label_index)
        loss = criterion(out.view(-1), train_labels)
        loss.backward()
        optimizer.step()

        # Refresh SMOTE every 10 epochs on current encoder embeddings
        if epoch % 10 == 0:
            try:
                with torch.no_grad():
                    z = model.encode(node_features, train_mp_edge_index)
                    src = z[train_edge_label_index[0]]
                    dst = z[train_edge_label_index[1]]
                    z_pairs = torch.cat([src, dst], dim=-1).cpu().numpy()
                z_res, y_res = smote.fit_resample(z_pairs, y_np)
                smote_data = (
                    torch.tensor(z_res, dtype=torch.float32).to(device),
                    torch.tensor(y_res, dtype=torch.float32).to(device),
                )
            except Exception:
                smote_data = None

        # SMOTE decoder pass using cached augmented embeddings
        if smote_data is not None:
            z_res_t, y_res_t = smote_data
            optimizer.zero_grad()
            out = model.lin1(z_res_t)
            out = torch.relu(out)
            out = model.lin2(out)
            loss = criterion(out.view(-1), y_res_t)
            loss.backward()
            optimizer.step()

        # Train on one randomly sampled augmented graph per epoch
        if augmented_data:
            aug_x, aug_mp = rng.choice(augmented_data)
            optimizer.zero_grad()
            out = model(aug_x, aug_mp, train_edge_label_index)
            loss = criterion(out.view(-1), train_labels)
            loss.backward()
            optimizer.step()


    # 8. Validation accuracy
    model.eval()
    with torch.no_grad():
        out = model(node_features, val_mp_edge_index, val_edge_label_index)
        pred = (torch.sigmoid(out) > 0.5).float().view(-1)
        acc = (pred == val_labels).sum().item() / val_labels.size(0)

    # 9. For inference, use the same MP edge index (directed or undirected per relation)
    all_pos_mp = mp_edge_index.to(device)

    class ModelWrapper:
        def __init__(self, model, node_features, mp_edge_index, mapping, device):
            self.model = model
            self.x = node_features
            self.mp_edge_index = mp_edge_index
            self.mapping = mapping
            self.device = device

        def predict_proba(self, edge):
            u, v = edge
            if u not in self.mapping or v not in self.mapping:
                return np.array([[0.5, 0.5]])

            edge_label_index = torch.tensor(
                [[self.mapping[u]], [self.mapping[v]]], dtype=torch.long
            ).to(self.device)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(self.x, self.mp_edge_index, edge_label_index)
                prob = torch.sigmoid(logits).item()

            return np.array([[1 - prob, prob]])

    return ModelWrapper(model, node_features, all_pos_mp, mapping, device), acc



def predict_and_ask(
    alpha, relation, B, L, NL, target_network, variables, **kwargs
):
    
    ACCURACY_THRESHOLD = kwargs.get('tau', 0.7)
    START_THRESHOLD = kwargs.get('k', 20)

    consecutive_failures = 0
    G = nx.DiGraph() if relation.directed else nx.Graph()

    E = []
    for conj in L:
        for c in sorted(conj, key=str):
            if c.relation == relation:
                E.append(c)

    G.add_nodes_from([int(v.name) for v in variables])
    for c in E:
        G.add_edge(*[int(v.name) for v in c.scope])

    Excluded = [
        c for c in NL
        if c.relation == relation and all(int(v.name) in G.nodes() for v in c.scope)
    ]

    if len(E) < START_THRESHOLD//2 or len(Excluded) < START_THRESHOLD//2:
        return

    start_time = time.perf_counter()
    model, accuracy = fit_gnn_link_predictor(G, E, Excluded, **kwargs)
    target_network.set_time(time.perf_counter() - start_time)

    
    if model is None or accuracy < ACCURACY_THRESHOLD:
        return

    sorted_B = sorted(list(B), key=str)
    Delta = [c for c in sorted_B if c.relation == relation and c not in E]

    S = [(c, model.predict_proba([int(v.name) for v in c.scope])[0][1]) for c in Delta]
    rec = sorted(S, key=lambda x: x[1])

    while rec and consecutive_failures < alpha:
        c_candidat, _ = rec.pop()

        if target_network.ask_req(c_candidat):
            L.append({c_candidat})
            G.add_edge(*[int(v.name) for v in c_candidat.scope])
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            NL.append(c_candidat)

        B.remove(c_candidat)


def ml_quacq(alpha, B, target_network, variables, logger=None, **kwargs):
    """
    Main loop for Node-based ML Quacq
    """
    writelog(logger, f"ml_quacq (node/GNN) started — |B|={len(B)}, variables={len(variables)}", level="info")
    writelog(logger, f"Using alpha: {alpha}", level="info")
    L, NL = [], []
    B_initial_size = len(B)
    pbar = tqdm.tqdm(total=len(B), desc="Size of B", unit="constraints")

    while True:
        example = quacq.GenerateExample(B, L, variables, seed=kwargs.get("seed", 42))

        # Stop condition
        if example is None:
            pbar.n = B_initial_size
            pbar.refresh()
            pbar.close()
            writelog(logger, f"ml_quacq (node/GNN) converged — learned {len(L)} constraints, |B| reduced by {B_initial_size - len(B)}", level="info")
            return L

        if target_network.ask(example):
            K_B = set([c for c in B if c.check(example) == False])
            B.difference_update(K_B)
            NL += sorted(list(K_B), key=str)
            writelog(logger, f"example accepted, pruned {len(K_B)} constraints from B", level="debug")
        else:
            writelog(logger, f"example rejected, running findScope + findC", level="debug")
            L_prev = list(L)
            scope = quacq.findScope(
                example, set(), {v.name for v in variables}, NL, B, target_network
            )
            quacq.findC(example, scope, L, NL, B, target_network, variables)

            try:
                for conj in L:
                    if conj not in L_prev:
                        for c in sorted(conj, key=str):
                            if c.relation.arity == 2:
                                writelog(logger, f"running GNN link predictor for relation {c.relation}", level="debug")
                                predict_and_ask(alpha, c.relation, B, L, NL, target_network, variables, **kwargs)
            except Exception as e:
                writelog(logger, f"ML step failed: {e}", level="warning")

        pbar.n = B_initial_size - len(B)
        pbar.refresh()



if __name__ == "__main__":

    from benchmarks.zebra import Model  
    instance = Model()

    variables=instance.variables
    B=instance.bais
    target=TargetNetwork(instance.constraints)
    alpha=3
    
    L = ml_quacq(alpha, B, target, variables)
    eq = equivalent(target.constraints, L, variables)

    if eq:
        stats=target.get_statistics()
        print(stats)
    else:
        print("Learned constraints are not equivalent to target constraints.")
    