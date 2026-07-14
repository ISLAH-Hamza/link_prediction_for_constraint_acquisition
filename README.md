# ML-QuAcq — Recommendation Queries for Constraint Acquisition via Link Prediction

## Title

**ML-QuAcq: Recommendation Queries for Constraint Acquisition via Link Prediction**

- **Repository:** https://github.com/ISLAH-Hamza/link_prediction_for_constraint_acquisition
- **Author:** Hamza ISLAH — islah.hamza1997@gmail.com
- **License:** MIT

---

## Description

**ML-QuAcq** is a research framework for **interactive constraint acquisition**: a learner discovers an unknown constraint network (a CSP) by asking an oracle yes/no questions. This project extends the classical **QuAcq** algorithm with **link prediction** — a graph machine-learning model that looks at the constraints learned so far, treats them as a graph, and *recommends* the constraints most likely to be missing. Recommending a constraint costs a single query instead of a full example-generation round, so the total number of oracle queries drops.

Constraint acquisition learns a CSP from an oracle. The learner keeps a **bias** `B` (every constraint it *might* learn) and a set `L` of confirmed constraints. It repeatedly builds an example that is consistent with `L` but violates something in `B`, and asks the oracle "is this a valid solution?". A **yes** prunes the violated candidates; a **no** triggers a binary search (`findScope` + `findC`) that isolates and confirms one real constraint. This is expensive — many queries per learned constraint. Our contribution: after each new constraint is confirmed, we build a graph whose nodes are variables and whose edges are the confirmed constraints of a given relation, train a **link-prediction model** on it, and use the model to **directly recommend** the most likely missing edges. Each recommendation is one cheap `ask_req` query, and it stops after `alpha` consecutive rejections. This is **ML-QuAcq**, and we provide two flavours of the link predictor (edge-feature based and GNN based).

This README walks a reviewer through **what was built, how it works, what we tested and why, and how to reproduce every result.**

### Table of Contents

1. [Title](#title)
2. [Description](#description)
3. [Dataset Information](#dataset-information)
4. [Code Information](#code-information)
5. [Usage Instructions](#usage-instructions)
6. [Requirements](#requirements)
7. [Methodology](#methodology)
8. [Testing — what we test and why](#testing--what-we-test-and-why)
9. [Citations](#citations)
10. [License & Contribution Guidelines](#license--contribution-guidelines)

---

## Dataset Information

This project does not ship an external data file; its "datasets" are **four constraint-satisfaction benchmarks** generated in code. Each benchmark is a `Model` dataclass (in `src/benchmarks/`) exposing three fields:

- `variables` — the CSP variables, each with a numeric domain `[lo, hi]`;
- `constraints` — the **ground-truth** target constraint network the learner must recover (used by the oracle);
- `bais` — the **bias set `B`**: every candidate constraint the learner may consider.

Variable names are stringified integers so they map cleanly to graph node IDs.

| Benchmark | File | Vars | Domain | Description |
|---|---|---|---|---|
| `zebra` | `benchmarks/zebra.py` | 25 | `[0,4]` | Einstein's Zebra puzzle: 5 attributes × 5 houses |
| `jigsaw` | `benchmarks/jigsaw.py` | 36 | — | 6×6 jigsaw-Sudoku |
| `rflap` | `benchmarks/rflap.py` | 25 | `[0,32]` | Radio-Link Frequency Assignment (all-different + interference distance constraints) |
| `random` | `benchmarks/random_bench.py` | 20 | `[0,20]` | randomly generated CSP with a **planted** solution (density-controlled) |

**Supported constraint types** (`src/core/constraints.py`, 18 classes):

| Category | Constraints |
|---|---|
| Unary (value) | `EqualVal`, `NotEqualVal`, `GreaterVal`, `LessVal`, `GreaterEqualVal`, `LessEqualVal` |
| Binary | `Equal`, `NotEqual`, `Greater`, `Less`, `GreaterEqual`, `LessEqual` |
| Distance | `DistEqual`, `DistNotEqual`, `DistGreater`, `DistLess`, `DistGreaterEqual`, `DistLessEqual` |

**Key vocabulary**

| Term | Meaning |
|---|---|
| **Variable** | named item with a numeric domain `[lo, hi]`, e.g. `x ∈ [0, 4]` |
| **Constraint** | rule over one or two variables, e.g. `x ≠ y`, `\|x − y\| = 1` |
| **Bias `B`** | every candidate constraint the learner may consider |
| **Learned `L`** | constraints confirmed by the oracle (list of conjunctions) |
| **`NL` / excluded** | constraints proven *not* in the target |
| **Target network / oracle** | ground truth; answers `ask` (is this example valid?) and `ask_req` (is this specific constraint real?) |
| **Example** | full assignment of values to all variables |
| **`alpha`** | consecutive-rejection budget for recommendations |
| **`tau`** | minimum validation accuracy to trust the trained model |
| **`k`** | minimum labelled edges before the model may be trained |

The data model lives in `src/core/base.py`: `Variable`, `Relation`, `Constraint` (abstract), and `TargetNetwork`. Note that equality/hashing is **directionality-aware** — a directed relation (e.g. `Greater`) distinguishes `(x, y)` from `(y, x)`, while an undirected one does not. This is central to how constraints are compared and deduplicated, and it is tested heavily.

---

## Code Information

The code is a Python package (`ml_quacq`) laid out under `src/`, installed in editable mode. Its top-level importable packages are `core`, `acquisition`, `benchmarks`, `utils` (the `src/` directory is the package root — see `pyproject.toml`).

```
link_prediction_for_constraint_acquisition/
├── src/
│   ├── __main__.py               # CLI entry point (python -m src)
│   ├── utils.py                  # logging + fixed_seed
│   ├── core/
│   │   ├── base.py               # Variable, Relation, Constraint, TargetNetwork
│   │   ├── constraints.py        # 18 concrete constraint classes
│   │   └── solve.py              # Solve + equivalent (enumeration-based)
│   ├── acquisition/
│   │   ├── baselines/
│   │   │   ├── quacq.py          # QuAcq: GenerateExample, findScope, findC
│   │   │   └── pquacq.py         # PQuAcq (Adamic–Adar)
│   │   └── ml_quacq/
│   │       ├── ml_quacq_e.py     # edge variant (MLP + structural features)
│   │       └── ml_quacq_n.py     # node variant (GIN link predictor)
│   └── benchmarks/               # zebra, jigsaw, rflap, random_bench
├── tests/                        # unit / convergence / reproducibility (see Testing)
├── notebooks/                    # main.ipynb — exploratory notebook
├── pyproject.toml
└── README.md
```

**The four algorithms.** All share the same QuAcq backbone (`src/acquisition/baselines/quacq.py`); they differ only in the **recommendation step** that runs after a constraint is learned:

| Algorithm | File | Recommender | Model |
|---|---|---|---|
| **QuAcq** (baseline) | `acquisition/baselines/quacq.py` | none | — |
| **PQuAcq** | `acquisition/baselines/pquacq.py` | Adamic–Adar link-prediction index | heuristic (no training) |
| **ML-QuAcq-Edge** | `acquisition/ml_quacq/ml_quacq_e.py` | MLP on hand-crafted edge features | `scikit-learn` `MLPClassifier` |
| **ML-QuAcq-Node** | `acquisition/ml_quacq/ml_quacq_n.py` | GNN on learned node embeddings | 2-layer **GIN** (PyTorch Geometric) |

**A reviewer's reading order**

1. **`src/core/base.py`** — the data model and, importantly, the directionality-aware constraint equality that everything relies on.
2. **`src/acquisition/baselines/quacq.py`** — the QuAcq backbone (`GenerateExample`, `findScope`, `findC`). Every other method reuses this loop.
3. **`src/acquisition/baselines/pquacq.py`** — the simplest recommendation step (Adamic–Adar), to see where ML plugs in.
4. **`src/acquisition/ml_quacq/ml_quacq_e.py`** then **`ml_quacq_n.py`** — the two contributions. Focus on `predict_and_ask` (the recommendation loop) and `fit_link_predictor` / `fit_gnn_link_predictor` (the online-training safeguards).
5. **`tests/`** — the convergence tests are the ground-truth correctness proof; the reproducibility tests justify the experimental methodology.

---

## Usage Instructions

### Installation

```bash
git clone https://github.com/ISLAH-Hamza/link_prediction_for_constraint_acquisition.git
cd link_prediction_for_constraint_acquisition

# create/activate a virtualenv
python -m venv .venv && source .venv/bin/activate

# install in editable mode — REQUIRED: this wires up the src/ package layout
pip install -e .
```

> The editable install is mandatory: `pyproject.toml` sets `package-dir = {"" = "src"}`, so both the source and the tests import as `from core.base import ...` (not `from src.core...`). Nothing resolves without it.

### Command line

The entry point is `src/__main__.py`, run as a module:

```bash
python -m src --benchmark <name> --method <algo> [options]
```

> Note: `__main__.py` fixes `PYTHONHASHSEED=0` and re-execs itself on first launch so that Python's hash randomisation cannot make runs non-deterministic. This is why reproducibility holds across processes.

| Argument | Values | Default | Applies to |
|---|---|---|---|
| `--benchmark` | `zebra`, `jigsaw`, `rflap`, `random` | *required* | all |
| `--method` | `quacq`, `pquacq`, `ml_quacq_e`, `ml_quacq_n` | *required* | all |
| `--alpha` | int | `4` | all but `quacq` |
| `--tau` | float | `0.7` | ML variants (accuracy gate) |
| `--k` | int | `20` | ML variants (cold-start threshold) |
| `--num_layers` | int | `2` | `ml_quacq_n` (GIN depth) |
| `--features` | comma list or `none` | `none` (= all) | ML variants (feature subset) |
| `--seed` | int | `42` | all |
| `--output_dir` | path | `./results` | all |

**Examples:**

```bash
# Baseline QuAcq on Zebra
python -m src --benchmark zebra --method quacq

# PQuAcq (Adamic–Adar) with alpha = 3
python -m src --benchmark zebra --method pquacq --alpha 3

# ML-QuAcq edge (MLP), only jaccard + degree features
python -m src --benchmark jigsaw --method ml_quacq_e --alpha 4 --features jaccard,degree

# ML-QuAcq node (GIN), 1 layer, custom cold-start threshold
python -m src --benchmark rflap --method ml_quacq_n --alpha 4 --num_layers 1 --k 30
```

### Output format

Each run writes one JSON file to `output_dir`, named
`{benchmark}_{method}_{seed}_{alpha}_{timestamp}.json`:

```json
{
    "query": 830,          // total oracle queries (ask + ask_req)
    "ask": 263,            // membership queries (full examples)
    "ask_req": 567,        // recommendation queries
    "yes": 192,            // recommendations accepted
    "no": 375,             // recommendations rejected
    "time": 1.63,          // max single link-predictor training time (s)
    "benchmark": "jigsaw",
    "method": "ml_quacq_e",
    "convergence": true,   // learned network == target network?
    "alpha": 3, "tau": 0.7, "k": 20, "num_layers": 2, "features": null,
    "seed": 0,
    "total_time": 76.42    // wall-clock for the whole run (s)
}
```

`convergence` is the correctness check: it comes from `equivalent()`, which enumerates up to 100 solutions of both the learned and target networks and confirms they accept/reject the same assignments (soundness **and** completeness).

### Python API

```python
from benchmarks.zebra import Model
from core.base import TargetNetwork
from core.solve import equivalent, Solve
from acquisition.ml_quacq.ml_quacq_e import ml_quacq   # edge variant

instance = Model()
target = TargetNetwork(instance.constraints)

# Run learning (tau/k/features/seed are optional kwargs)
L = ml_quacq(alpha=4, B=instance.bais, target_network=target,
             variables=instance.variables, seed=42)

# Correctness + solve + stats
print(equivalent(target.constraints, L, instance.variables))  # True if equivalent
print(Solve(L, instance.variables))                           # a satisfying assignment
print(target.get_statistics())
# {'query':..., 'ask':..., 'ask_req':..., 'yes':..., 'no':..., 'time':...}
```

The node/GIN variant has the same signature:

```python
from acquisition.ml_quacq.ml_quacq_n import ml_quacq
L = ml_quacq(alpha=4, B=instance.bais, target_network=target,
             variables=instance.variables, num_layers=2, seed=42)
```

---

## Requirements

- **Python ≥ 3.8** (developed and tested on 3.11).

Dependencies are installed automatically from `pyproject.toml` by `pip install -e .`:

| Package | Used for |
|---|---|
| `ortools` | CP-SAT solver in `GenerateExample`, `findC`, `Solve`, equivalence checking |
| `networkx` | constraint-graph construction and structural features |
| `scikit-learn` | MLP, SMOTE, `StandardScaler` (edge variant) |
| `torch`, `torch_geometric` | GIN model (node variant) |
| `infomap` | community-detection feature (edge variant) |
| `imblearn` | SMOTE oversampling |
| `tqdm` | progress bars |
| `matplotlib` | plotting utilities |
| `pytest` | test suite |
| `black` | code formatting |
| `ipykernel` | notebook support |

---

## Methodology

All four algorithms share the same QuAcq backbone and differ only in the **recommendation step** that runs after a constraint is learned.

### QuAcq (baseline)

The classical loop:

1. `GenerateExample` — build an assignment that satisfies `L` and violates ≥ 1 constraint in `B` (solved with OR-Tools CP-SAT).
2. `ask` the oracle.
3. **Yes** → remove every constraint in `B` the example violates.
4. **No** → `findScope` (recursive binary search over the variables) narrows down which variables are responsible, then `findC` isolates the exact constraint (or conjunction) and moves it into `L`.
5. Terminate when no violating example can be generated (`B` is exhausted or entailed by `L`).

### PQuAcq

QuAcq **plus** a proactive step. After learning a binary constraint, it builds the constraint graph and scores every candidate edge in `B` with the **Adamic–Adar** index; it repeatedly recommends the highest-scoring candidate until `alpha` consecutive rejections. This is our non-ML recommendation baseline.

### ML-QuAcq-Edge (`ml_quacq_e`)

The recommender is a trained **MLP**. For each candidate edge `(u, v)` it builds a feature vector:

| Feature | Size | Meaning |
|---|---|---|
| `degree` | 4 | in/out degree of `u` and of `v` |
| `jaccard` | 1 | neighbourhood overlap |
| `resource_allocation` | 1 | inverse-degree-weighted common neighbours |
| `infomap` | 1 | 1 if `u` and `v` are in the same Infomap community, else 0 |

The set of features is configurable (`--features`; see `FEATURE_REGISTRY` in `ml_quacq_e.py`), which makes leave-one-out feature ablations straightforward.

### ML-QuAcq-Node (`ml_quacq_n`)

The recommender is a **Graph Isomorphism Network (GIN)**. Instead of hand-crafted edge features it learns **node embeddings** by message passing, then predicts an edge by concatenating its two endpoint embeddings and passing them through a small MLP head. Node features:

| User-facing name | Expands to | Meaning |
|---|---|---|
| `degree` | in-degree + out-degree | local structure (normalised) |
| `betweenness` | betweenness | global centrality (normalised) |
| `closeness` | in-closeness + out-closeness | global centrality (normalised) |

`--num_layers` controls the number of GIN layers (receptive-field size).

### Safeguards shared by both ML variants

The model is trained *online* on very little data, so we protect against garbage recommendations:

- **Cold-start guard (`k`, default 20):** don't even train until there are at least `k//2` positive **and** `k//2` negative example edges for that relation.
- **Accuracy gate (`tau`, default 0.7):** train with an 80/20 validation split; if validation accuracy `< tau`, throw the model away and skip recommending.
- **Class balancing:** SMOTE oversamples the minority class before training.
- **Data augmentation:** when fewer than 100 labelled edges exist, train on 10 perturbed copies of the graph (`graph_drop_edges`) to reduce overfitting.
- **Alpha stop (`alpha`):** stop recommending after `alpha` consecutive oracle rejections.

---

## Testing — what we test and why

```bash
pytest tests/                                   # everything
pytest tests/test_core/                         # just the data model (fast)
pytest tests/test_quacq/test_quacq_acquisition.py::TestGenerateExample
black src/ tests/                               # formatting
```

The suite is organised so that every algorithm has the **same three kinds of test**, which together answer three distinct questions: *are the pieces correct?* (unit), *does the whole thing learn the right network?* (convergence), and *is it deterministic?* (reproducibility). Plus a sanity check that the benchmarks themselves are solvable.

```
tests/
├── test_core/test_base.py                      # data model (Variable/Relation/Constraint/TargetNetwork)
├── test_quacq/          {acquisition, convergence, reproducebility}
├── test_pquacq/         {acquisition, convergence, reproducebility}
├── test_ml_quacq/edge/  {acquisition, convergence, reproducebility}
├── test_ml_quacq/node/  {acquisition, convergence, reproducebility}
└── test_benchmarks/test_benchmark_validation.py
```

### 1. Unit / acquisition tests — *are the building blocks correct?*

Fast, isolated tests on individual functions.

| Module | What it verifies (and why it matters) |
|---|---|
| `test_core/test_base.py` | `Variable` validation (rejects bad names/domains), `Relation` equality & hashing, and — crucially — **directionality-aware** `Constraint` equality/hashing (directed constraints distinguish scope order, undirected ones don't). Also `TargetNetwork` query counting and human-mode input. If these are wrong, every downstream set operation on `B`/`L`/`NL` breaks silently. |
| `test_quacq/test_quacq_acquisition.py` | `GenerateExample` guarantees: returns all variable keys, violates ≥1 `B` constraint, satisfies all `L`, returns `None` when `B` is empty or entailed, **never mutates `B`**, respects domains. Parametrised over all distance constraint types. |
| `test_pquacq/test_pquacq_acquisition.py` | Same `GenerateExample` guarantees, plus the **Adamic–Adar** score (0 for disconnected nodes, increases with common neighbours, skips degree-1 neighbours) and the recommendation loop (α=0 → no proactive asks; higher α → more; constant-0 vs constant-1 score functions). |
| `test_ml_quacq/edge/…_acquisition.py` | Feature vector **shape (7-dim)** and correctness (Jaccard ∈ [0,1], RA ≥ 0, infomap flag, degree slots, zeros for missing nodes); `graph_drop_edges` doesn't mutate the input and drops the right count; `fit_link_predictor` returns a usable model with accuracy ∈ [0,1] and the right `predict_proba` shape. |
| `test_ml_quacq/node/…_acquisition.py` | `get_node_feature_matrix` shape/type and directed/undirected degree equivalence; `convert_to_pyg_data` mapping coverage, contiguity and edge-index shape; `LinkPredictor` forward/encode/decode shapes; `fit_gnn_link_predictor` returns `None` when nodes don't match, and `predict_proba` sums to 1 (and returns `[0.5, 0.5]` for unknown nodes). Also that `predict_and_ask` **does nothing** when data is insufficient or below the accuracy gate. |

### 2. Convergence tests — *does it learn the correct network?*

End-to-end: run the full acquisition loop on real benchmarks and assert
`equivalent(target, L, variables)` — i.e. the learned network accepts and rejects exactly the same assignments as the ground truth. This is the property that actually matters for a constraint-acquisition algorithm.

| Algorithm | Benchmarks |
|---|---|
| QuAcq | Zebra, Jigsaw |
| PQuAcq | Zebra, Jigsaw |
| ML-QuAcq Edge | Zebra, Jigsaw |
| ML-QuAcq Node | Zebra, Jigsaw |

> These run the real solver over thousands of candidate constraints and can take several minutes (Zebra especially).

### 3. Reproducibility tests — *is it deterministic?*

Run each algorithm **twice with the same seed** (over 3 seeds) and assert the runs are identical:

- `len(L1) == len(L2)` — same constraints learned
- `ask_counter` equal — same number of oracle queries
- for the proactive methods (PQuAcq, ML-QuAcq): `req_yes` and `req_no` equal — same recommendations accepted/rejected

Determinism is a hard requirement here because the algorithms are stochastic (random example generation, SMOTE, GNN init); without the seeding discipline in `fixed_seed` + `PYTHONHASHSEED=0` the experiments would not be comparable.

### 4. Benchmark validation — *are the benchmarks themselves well-formed?*

`test_benchmark_validation.py` confirms each benchmark's ground-truth constraint set is **solvable** (the CP solver finds at least one assignment) for Zebra, Jigsaw and RFLAP. A benchmark with no solution would make every other result meaningless.

---

## Citations


This work builds on the constraint-acquisition literature, in particular QuAcq (partial queries) and the recommendation-query setting:

```bibtex
@inproceedings{bessiere2013constraint,
  title     = {Constraint Acquisition via Partial Queries},
  author    = {Bessiere, Christian and Coletta, Remi and Hebrard, Emmanuel and
               Katsirelos, George and Lazaar, Nadjib and Narodytska, Nina and
               Quimper, Claude-Guy and Walsh, Toby},
  booktitle = {Proceedings of the 23rd International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {475--481},
  year      = {2013}
}

@inproceedings{daoudi2016constraint,
  title     = {Constraint Acquisition with Recommendation Queries},
  author    = {Daoudi, Abderrazak and Mechqrane, Younes and Bessiere, Christian and
               Lazaar, Nadjib and Bouyakhf, El-Houssine},
  booktitle = {Proceedings of the 25th International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {720--726},
  year      = {2016}
}
```

---

## License & Contribution Guidelines

### License

This project is released under the **MIT License** (see the `license` field in `pyproject.toml`).

### Contribution Guidelines

Contributions are welcome via GitHub issues and pull requests at
https://github.com/ISLAH-Hamza/link_prediction_for_constraint_acquisition.

- **Branch & PR:** fork the repo, create a feature branch, and open a pull request against `main` describing the change.
- **Formatting:** run `black src/ tests/` before committing — the project uses `black` as its formatter.
- **Tests:** add or update tests under `tests/` and ensure `pytest tests/` passes. New algorithms should follow the existing three-part pattern (acquisition / convergence / reproducibility). New benchmarks should include a solvability check in `test_benchmarks/`.
- **Determinism:** any stochastic code must respect the seeding discipline (`utils.fixed_seed` + `PYTHONHASHSEED=0`) so reproducibility tests keep passing.

**Author:** Hamza ISLAH — islah.hamza1997@gmail.com