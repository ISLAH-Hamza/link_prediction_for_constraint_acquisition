import random, infomap, time, tqdm
import numpy as np
import networkx as nx

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


from acquisition.baselines import quacq
from core.solve import equivalent
from core.base import TargetNetwork
from utils import writelog




def _feat_jaccard(G, u, v, **kwargs):
    G_und = G.to_undirected() if G.is_directed() else G
    preds = list(nx.jaccard_coefficient(G_und, [(u, v)]))
    return [preds[0][2] if preds else 0.0]


def _feat_resource_allocation(G, u, v, **kwargs):
    G_und = G.to_undirected() if G.is_directed() else G
    try:
        preds = list(nx.resource_allocation_index(G_und, [(u, v)]))
        return [preds[0][2] if preds else 0.0]
    except Exception:
        return [0.0]


def _feat_infomap(G, u, v, **kwargs):
    modules = kwargs.get('infomap_modules')
    if modules is None:
        seed = max(1, int(kwargs.get('seed', 42)))
        im = infomap.Infomap(f"--two-level --silent --seed {seed}")
        for node in G.nodes:
            im.addNode(int(node))
        for e in G.edges:
            im.addLink(int(e[0]), int(e[1]))
        im.run()
        modules = im.getModules()
    try:
        return [1.0 if modules[int(u)] == modules[int(v)] else 0.0]
    except Exception:
        return [0.0]


def _feat_degree(G, u, v, **kwargs):
    if G.is_directed():
        return [G.out_degree(u), G.in_degree(u), G.out_degree(v), G.in_degree(v)]
    return [G.degree(u), G.degree(u), G.degree(v), G.degree(v)]


FEATURE_REGISTRY = {
    "jaccard": (_feat_jaccard, 1),
    "resource_allocation": (_feat_resource_allocation, 1),
    "infomap": (_feat_infomap, 1),
    "degree": (_feat_degree, 4),
}

DEFAULT_FEATURES = ["jaccard", "resource_allocation", "infomap", "degree"]


def compute_features(G, edge, features=None, **kwargs) -> np.ndarray:
    """
    Computes a feature vector for a given edge (u, v).

    Args:
        G (nx.Graph): The graph.
        edge (tuple): (u, v) node names/IDs.
        features (list | None): which features to compute.
            Defaults to DEFAULT_FEATURES. Names must be keys of FEATURE_REGISTRY.
        **kwargs: Additional data (e.g., infomap_modules, seed).

    Returns:
        np.ndarray: 1D array of feature values.
    """
    if features is None:
        features = DEFAULT_FEATURES

    total_size = sum(FEATURE_REGISTRY[name][1] for name in features)
    u, v = edge

    if not G.has_node(u) or not G.has_node(v):
        return np.zeros(total_size)

    vec = []
    for name in features:
        fn, _ = FEATURE_REGISTRY[name]
        vec.extend(fn(G, u, v, **kwargs))

    return np.array(vec)



def graph_drop_edges(G, n=1, seed=None):
    """
    Returns a copy of G with n randomly removed edges.

    Arguments:
        -> G (networkx.Graph) : the graph to be augmented
        -> n (int)            : the number of edges to remove, default is 1
        -> seed (int|None)    : random seed for reproducibility

    Returns:
        <- graph (networkx.Graph): the new graph with the removed edges
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


def _default_classifier(seed):
    return MLPClassifier(
        hidden_layer_sizes=(50, 10),
        max_iter=500,
        alpha=0.1,
        activation="relu",
        solver="adam",
        random_state=seed,
        learning_rate="adaptive",
        learning_rate_init=0.1,
    )


def fit_link_predictor(G, positiveE, negativeE, seed=42, features=None, classifier=None):

    mlp = classifier(seed) if classifier is not None else _default_classifier(seed)

    positive, negative = [], []

    if len(positiveE) + len(negativeE) < 100:
        for i in range(10):
            graph = graph_drop_edges(G, seed=seed + i + len(positiveE))
            positive += [
                [*compute_features(graph, [int(v.name) for v in item.scope], features=features, seed=seed), 1]
                for item in positiveE
            ]
            negative += [
                [*compute_features(graph, [int(v.name) for v in item.scope], features=features, seed=seed), 0]
                for item in negativeE
            ]

    positive += [
        [*compute_features(G, [int(v.name) for v in item.scope], features=features, seed=seed), 1]
        for item in positiveE
    ]
    negative += [
        [*compute_features(G, [int(v.name) for v in item.scope], features=features, seed=seed), 0]
        for item in negativeE
    ]

    data = np.array(positive + negative, dtype=float)
    X, y = data[:, :-1], data[:, -1]

    smote = SMOTE(random_state=seed)

    # --- Validation Step ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_res = scaler.fit_transform(X_train_res)
        X_test_n = scaler.transform(X_test)

        mlp.fit(X_train_res, y_train_res)
        y_pred = mlp.predict(X_test_n)
        accuracy = accuracy_score(y_test, y_pred)

    except ValueError:
        return None, 0.0

    # --- Final Training Step (train on full data) ---
    try:
        X_res, y_res = smote.fit_resample(X, y)
    except Exception:
        X_res, y_res = X, y

    scaler = StandardScaler()
    X_res_n = scaler.fit_transform(X_res)

    mlp.fit(X_res_n, y_res)

    class ModelWrapper:
        def __init__(self, model, graph, scaler, seed, features):
            self.model = model
            self.graph = graph
            self.scaler = scaler
            self.seed = seed
            self.features = features

        def predict_proba(self, edge):
            edge_features = compute_features(
                self.graph, edge, features=self.features, seed=self.seed
            ).reshape(1, -1)
            edge_features_n = self.scaler.transform(edge_features)
            return self.model.predict_proba(edge_features_n)

    return ModelWrapper(mlp, G, scaler, seed, features), accuracy



def predict_and_ask(
    alpha, relation, B, L, NL, target_network, variables, **kwargs
):
    consecutive_failures = 0
    G = nx.DiGraph() if relation.directed else nx.Graph()

    ACCURACY_THRESHOLD = kwargs.get('tau', 0.7)
    START_THRESHOLD = kwargs.get('k', 20)
    E = []
    for conj in L:
        for c in sorted(conj, key=str):
            if c.relation == relation:
                E.append(c)

    G.add_nodes_from([int(v.name) for v in variables])
    for c in E:
        G.add_edge(*[int(v.name) for v in c.scope])

    Excluded = [
        c
        for c in NL
        if c.relation == relation and all(int(v.name) in G.nodes() for v in c.scope)
    ]

    if len(E) < START_THRESHOLD//2 or len(Excluded) < START_THRESHOLD//2:
        return

    start_time = time.perf_counter()
    model, accuracy = fit_link_predictor(
        G,
        E,
        Excluded,
        seed=kwargs.get('seed', 42),
        features=kwargs.get('features'),
        classifier=kwargs.get('classifier'),
    )
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
    Learn constraint network using both recommendation and membership queries.

    Arguments:
     -> alpha (int)             : number of allowed consecutive failures
     -> B (set)                 : bias of the problem
     -> target_network (object) : oracle
     -> variables (list)        : variables of the problem
     -> link_predictor          : function to train the link prediction model

    Returns:
     <- L (list): learned constraint network
    """
    
    writelog(logger, f"ml_quacq (edge/MLP) started — |B|={len(B)}  variables={len(variables)}", level="info")
    writelog(logger, f"Using alpha: {alpha}", level="info")
    L, NL = [], []
    B_initial_size = len(B)
    pbar = tqdm.tqdm(total=len(B), desc="Size of B", unit="constraints")
    while True:

        example = quacq.GenerateExample(B, L, variables, seed=kwargs.get("seed", 42))
        if example is None:
            pbar.n = B_initial_size
            pbar.refresh()
            pbar.close()
            writelog(logger, f"ml_quacq (edge/MLP) converged — learned {len(L)} constraints, |B| reduced by {B_initial_size - len(B)}", level="info")
            return L

        if target_network.ask(example):
            K_B = set([c for c in B if c.check(example) == False])
            B.difference_update(K_B)
            NL += sorted(list(K_B), key=str)
            writelog(logger, f"example accepted, pruned {len(K_B)} constraints from B", level="debug")
        else:
            writelog(logger, "example rejected, running findScope + findC", level="debug")
            L_prev = list(L)
            scope = quacq.findScope(example, set(), {v.name for v in variables}, NL, B, target_network,logger=logger)
            quacq.findC(example, scope, L, NL, B, target_network, variables, logger=logger)

            for conj in L:
                if conj not in L_prev:
                    for c in sorted(conj, key=str):
                        if c.relation.arity == 2:
                            writelog(logger, f"running link predictor for relation {c.relation}", level="debug")
                            predict_and_ask(alpha, c.relation, B, L, NL, target_network, variables, logger=logger, **kwargs)

        pbar.n = B_initial_size - len(B)
        pbar.refresh()
