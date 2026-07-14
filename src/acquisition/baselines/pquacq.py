import networkx as nx
import numpy as np
from tqdm import tqdm

# local imports
import acquisition.baselines.quacq as quacq
from core.base import TargetNetwork
from core.solve import Solve
from utils import writelog





def adamic_adar(graph, edge) -> float:
    """
    calculate the adamic adar index of an edge in a graph
    Arguments:
        -> graph (nx.Graph) : networkx graph
        -> edge (list)      : the list of nodes (edge)
    Output:
        <- score (float): The adamic adar score of the edge
    """

    common_neighbors = list(nx.common_neighbors(graph, edge[0], edge[1]))
    score = 0
    for neighbor in common_neighbors:
        degree = graph.degree(neighbor)
        if degree > 1:
            score += 1 / (np.log(degree))
    return score


def predict_and_ask(r, score, alpha, B, L, target_network) -> None:
    """
    predict missing constraint based on the constraints network structure and suggest it into to user/oracle
    Arguments:
        -> r (relation) :
        -> score (function) :
        -> alpha (int)  :
        -> B (set)  :
        -> L (set)  :
        -> target_network (instance) :
    """
    ## initialization
    No = 0
    G = nx.Graph()
    E = []
    for conj in L:
        for c in sorted(conj, key=str):
            if c.relation == r:
                E.append(c)
    ## Constraint a Graph from the element of E
    for c in E:
        G.add_edge(*[v.name for v in c.scope])


    sorted_B = sorted(list(B), key=str)
    Delta = [c for c in sorted_B if c.relation == r and c not in E]
    
    while len(Delta) > 0 and No < alpha:
        S = []
        for c in Delta:

            G.add_nodes_from([v.name for v in c.scope])
            S.append([c, score(G, [v.name for v in c.scope])])
        rec = max(S, key=lambda x: x[-1])

        if rec[-1] == 0:
            break

        c_candidat = rec[0]

        if target_network.ask_req(c_candidat):
            L.append({c_candidat})
            G.add_edge(*[v.name for v in c_candidat.scope])
            No = 0
        else:
            No += 1

        Delta = [c for c in Delta if c != c_candidat]
        B.remove(c_candidat)




def pquacq(alpha, score, B, target_network, variables, logger=None, **kwargs):
    """
    PQuAcq Implementation with logging.
    """
    
    writelog(logger, f"PQuAcq started — |B|={len(B)}, variables={len(variables)}")
    writelog(logger, f"Using score function: {score.__name__}", level="info")
    writelog(logger, f"Using alpha: {alpha}", level="info")

    L = []
    Ex=[]
    B_initial_size = len(B)
    pbar = tqdm(total=len(B), desc="Size of B", unit="constraints")
    while True:
        example = quacq.GenerateExample(B, L, variables, seed=kwargs.get("seed", 42))
        if example is None:
            pbar.n = B_initial_size
            pbar.refresh()
            pbar.close()
            writelog(logger, f"PQuAcq converged — learned {len(L)} constraints, |B| reduced by {B_initial_size - len(B)}")
            return L

        if target_network.ask(example):
            K_B = {c for c in B if not c.check(example)}
            B.difference_update(K_B)
        else:
            scope = quacq.findScope(
                example, set(), {v.name for v in variables}, Ex, B, target_network,logger=logger
            )
            quacq.findC(example, scope, L, Ex, B, target_network, variables, logger=logger)

            last_learned_set = sorted(list(L)[-1], key=str)
            
            for c in last_learned_set:
                if c.relation.arity == 2:
                    predict_and_ask(c.relation, score, alpha, B, L, target_network)
            

        pbar.n = B_initial_size - len(B)
        pbar.refresh()




if __name__ == "__main__":

    from benchmarks.zebra import Model as Zebra
    instance = Zebra()
    variables=instance.variables
    B=instance.bais
    target=TargetNetwork(instance.constraints)
    L = pquacq(alpha=3, score=adamic_adar, B=B, target_network=target, variables=variables)
    sol=Solve(L, variables)
    print(sol)
    print("asks:",target.ask_counter,"yes:", target.req_yes, "no:", target.req_no)