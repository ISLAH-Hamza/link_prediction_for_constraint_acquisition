import itertools
from ortools.sat.python import cp_model
from tqdm import tqdm
from core.solve import Solve
from core.base import TargetNetwork
from utils import writelog
# ======================================================
# ================== GenerateExample  ==================
# ======================================================

def GenerateExample(B,L,vars,seed = 0):
    """
        Generate an example that satisfies the constraints in L while rejecting at least one constraint in B.
        args:
            B       : The Basis represent all possible constraint that could be used to model the problem.
            L       : The list of constraints to satisfy.
            vars    : The list of variables to use in the constraints.
            seed    : The random seed for the OR-Tools solver.
    """
    m=cp_model.CpModel()
    bools=[]
    variables={v.name:m.NewIntVar(*v.domain,v.name) for v in vars}

    for conj in L:
        for c in conj:
             c.to_ortools(m,variables=variables)

    for c in B:
        bools.append(m.NewBoolVar(f'b{len(bools)}'))
        c.to_ortools_boolean(bools[-1],m,variables)
    

    m.Add(sum(bools) != len(B))

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = seed
    status = solver.Solve(m)


    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        sol={i: solver.Value(variables[i])for i in variables.keys()}
        return sol
    


# ======================================================
# ==================== FindSCope  ======================
# ======================================================

def findScope(example,R,Y,Ex,B,target_network,logger=None):
    """
        a recursive process to find the scope of variables in example that make the answer of the user negative see more information in the original paper.
        args:
            example         : The example to analyze.
            R               : The set of relevant variables.
            Y               : The set of variables to consider.
            Ex              : The set of excluded constraints.
            target_network  : The target network (which model the user's oracle/answers).
            logger          : An optional logger to log information.
    """
    e_R={key:val for key,val in example.items() if key in R }
    K_B=set(c for c in B if c.check(e_R)==False)

    if len(K_B)>0:
        if target_network.ask(e_R):
            B.difference_update(K_B)
            Ex.extend(K_B)
            writelog(logger, f"findScope: pruned {len(K_B)} constraints from B", level="debug")
        else: return set()

    if len(Y)==1:
        writelog(logger, f"findScope: scope found -> {Y}", level="debug")
        return Y
    Y_list = sorted(Y)
    n = len(Y_list) // 2
    Y1, Y2 = set(Y_list[:n]), set(Y_list[n:])
    e_R_Y1={key:val for key,val in example.items() if key in set(R)|Y1}
    e_R_Y={key:val for key,val in example.items() if key in set(R)|Y}
    
    if set(c for c in B if c.check(e_R_Y1)==False)==set(c for c in B if c.check(e_R_Y)==False):
        S1=set()
    else:
        S1=findScope(example,set(R)|Y1,Y2,Ex,B,target_network,logger=logger)

    e_R_S1={key:val for key,val in example.items() if key in set(R)|S1}
    if set(c for c in B if c.check(e_R_S1)==False)==set(c for c in B if c.check(e_R_Y)==False):
        S2=set()
    else:
        S2=findScope(example,set(R)|S1,Y1,Ex,B,target_network,logger=logger)

    return S1|S2



# =================================================================
# ================ findC and it's helper functions ================
# =================================================================

def checkConj(s):
    """
        helper subprosess that will be used with findC, it checks if a conjunction of constraints is satisfiable.
        args:
            s : The conjunction of constraints to check.
    """
    s=list(s)

    if len(s)==0:
        return False 
    
    elif any(set(v.name for v in i.scope) != set(v.name for v in s[0].scope) for i in s):
        return False

    else:
        
        m=cp_model.CpModel()
        variables=dict()

        ### Creating Cp model variables and constraints
        for c in s:
            for v in c.scope:
                if v.name not in variables.keys():
                    variables[v.name]=m.NewIntVar(*v.domain,name=v.name)

        for c in s: c.to_ortools(m,variables)
        
        sovler=cp_model.CpSolver()
        # Single worker avoids spawning a thread pool for each of the many
        # tiny models that checkConj is called with in join_operation.
        sovler.parameters.num_workers = 1
        status=sovler.Solve(m)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return True

        else: return False


def join_operation(S,S_prime):
    """
        the join operation used in findC procedure see more information in the original paper.
        it will join to list of constraints S and S' and return a new list of constraints.
        args:
            S       : The first list of constraints.
            S_prime : The second list of constraints.
    """
    Output=[]
    for s1,s2 in  itertools.product(S,S_prime):
        s=s1|s2
        if s not in Output and checkConj(s)==True: Output.append(s)

    return Output



def FindEprime(L,Y,Delta,vars,seed=42,logger=None,max_iterations=1000000) -> dict:
    """
        find an example that satisfy the constraints in L and reject at least one conjunction in Delta.
        args:
            L              : The list of constraints to satisfy.
            Y              : The set of variables to consider.
            Delta          : set of candidate constraints.
            vars           : The set of all variables.
            seed           : The random seed for the OR-Tools solver.
            logger         : An optional logger to log information.
            max_iterations : Maximum number of solver calls before giving up.
    """
    m=cp_model.CpModel()

    ### creating ortools variables
    variables={v.name:m.NewIntVar(v.domain[0],v.domain[1],name=v.name) for v in vars}

    for conj in L:
        for c in conj:
            if all(v.name in Y for v in c.scope):
                c.to_ortools(m,variables)

    solver=cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = seed
    for _ in range(max_iterations):
        status=solver.Solve(m)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution={v:solver.Value(variables[v]) for v in Y}

            K_Delta=[conj for conj in Delta  if any(c.check(solution) is False for c in conj) ]

            if len(K_Delta)<len(Delta) and len(K_Delta)>0:
                return solution
            else:
                m.AddForbiddenAssignments([variables[v] for v in solution.keys()],
                                          [tuple(solution.values())]
                                          )
        else:
            return
    

def findC(example,Y,L,Ex,B,target_network,variables,logger=None):
    

    """
        find the conjunction of constraints that make the example negative see more information in the original paper.
        args:
            example         : The (negative)example to analyze.
            Y               : The set of variables to consider.
            L               : The list of constraints to satisfy.
            B               : The Bais.
            target_network  : The target network (which model the user's oracle/answers).
            variables       : The set of all variables.
            logger          : An optional logger to log information.
    """
    B_Y=set([c for c in B if all(v.name in Y for v in c.scope)])
    
    Delta=[{item} for item in B_Y]
    K_Delta=[conj for conj in Delta  if any(c.check(example)==False for c in conj) ]
    Delta=join_operation(Delta,K_Delta)
    if not Delta:
        return 
    
    while True:
        eprime=FindEprime(L,Y,Delta,variables)
        if eprime == None:
            L.append((Delta[0]))
            B.difference_update(Delta[0])
            writelog(logger, f"findC: learned constraint {sorted(Delta[0], key=str)}", level="info")
            return
        else:
            K_DeltaEprime=[conj for conj in Delta  if any(c.check(eprime)==False for c in conj)]
            if target_network.ask(eprime)==True:
                Delta=[conj for conj in Delta if conj not in K_DeltaEprime]
                Ex.extend(set(c for c in B if c.check(eprime)==False))
                B.difference_update(set(c for c in B if c.check(eprime)==False))
               
            else:
                S=findScope(eprime,set(),Y,Ex,B,target_network,logger=logger)
                if all(i in Y for i in S) and len(S)<len(Y):
                    findC(eprime,S,L,Ex,B,target_network,variables,logger=logger)
                else:
                    Delta=join_operation(Delta,K_DeltaEprime)




def QuAcq(B,target_network,variables,logger=None,**kwargs):
    """
        The main QuAcq algorithm to learn the target network using the basis B and the given variables.
        args:
            B               : The Basis represent all possible constraint that could be used to model the problem.
            variables       : The set of all variables.
            target_network  : The target network (which model the user's oracle/answers).
            logger          : An optional logger to log information.
    """


    L = []
    Ex= []
    B_initial_size = len(B)
    writelog(logger, f"QuAcq started — |B|={B_initial_size}, variables={len(variables)}", level="info")
    pbar = tqdm(total=len(B), desc="Size of B", unit="constraints")

    while True:

        example = GenerateExample(B, L, variables, seed=kwargs.get("seed", 42))
        if example is None:
            pbar.n = B_initial_size - len(B)
            pbar.refresh()
            pbar.close()
            writelog(logger, f"QuAcq converged — learned {len(L)} constraints, |B| reduced by {B_initial_size - len(B)}", level="info")
            return L

        if target_network.ask(example):
            K_B = {c for c in B if not c.check(example)}
            B.difference_update(K_B)
            Ex.extend(K_B)
            writelog(logger, f"QuAcq: example accepted, pruned {len(K_B)} constraints from B", level="debug")
        else:
            writelog(logger, f"QuAcq: example rejected, running findScope + findC", level="debug")
            scope = findScope(example, set(), {v.name for v in variables}, Ex, B, target_network, logger=logger)
            findC(example, scope, L, Ex, B, target_network, variables, logger=logger)

        pbar.n = B_initial_size - len(B)
        pbar.refresh()


if __name__ == "__main__":
    
        from benchmarks.zebra import Model as Zebra
        instance = Zebra()
        variables=instance.variables
        B=instance.bais
        target=TargetNetwork(instance.constraints)
        L = QuAcq(B, target, variables)
        sol=Solve(L, variables)
        print(sol)