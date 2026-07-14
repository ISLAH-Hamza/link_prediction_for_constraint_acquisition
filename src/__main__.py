import argparse,os,json,time,logging,sys

# PYTHONHASHSEED must be fixed *before* the interpreter starts hashing objects;
# if it wasn't set in the environment we set it and re-exec so that Python's
# hash randomisation is deterministic across runs.
if os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# setup_file_logging must be called before any other local import so that the
# FileHandler is attached to the 'ml_quacq' root logger before child loggers
# are created at module level in the imported packages.
from utils import fixed_seed, setup_logger, writelog

from benchmarks import zebra, jigsaw, rflap, random_bench
from acquisition.baselines import quacq, pquacq
from acquisition.ml_quacq import ml_quacq_e, ml_quacq_n
from core.base import TargetNetwork
from core.solve import equivalent

_BENCHMARK_MAPPING = {
        "zebra": zebra.Model,
        "jigsaw": jigsaw.Model,
        "rflap": rflap.Model,
        "random": random_bench.Model,
    }

def main(benchmark, method,**kwargs):

   
    logger = setup_logger(to_console=False)
    writelog(logger, f"Starting — benchmark={benchmark}, method={method}", level="info")
    seed= kwargs.get("seed", 42)
    fixed_seed(seed)
    writelog(logger, f"Random seed set to {kwargs.get('seed', 42)}", level="info")
    
    model = _BENCHMARK_MAPPING[benchmark]()
    target = TargetNetwork(model.constraints)
    variables = model.variables  
    B=model.bais
    alpha = kwargs.get("alpha", 1)
    

    starting_time = time.time()
    if method == "quacq":
        L=quacq.QuAcq(B, target, variables,logger=logger,seed=seed)

    elif method == "pquacq":
        L=pquacq.pquacq(alpha, pquacq.adamic_adar, B, target, variables, logger=logger, seed=seed)

    elif method == "ml_quacq_e":
        features = kwargs.get("features", None)
        tau = kwargs.get("tau", 0.7)
        k = kwargs.get("k", 20)

        L=ml_quacq_e.ml_quacq(alpha, B, target, variables, logger=logger,seed=seed, features=features, tau=tau, k=k)

    elif method == "ml_quacq_n":
        features = kwargs.get("features", None)
        tau = kwargs.get("tau", 0.7)
        k = kwargs.get("k", 20)
        num_layers = kwargs.get("num_layers", 2)
        L=ml_quacq_n.ml_quacq(alpha, B, target, variables, logger=logger,seed=seed,features=features, tau=tau, k=k, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown learning method: {method}")
    
    endint_time = time.time()

    is_equivalent = equivalent(target.constraints,L, variables)
    # save resutls to output_dir if needed
    os.makedirs(kwargs.get("output_dir", "./results"), exist_ok=True)
    print(kwargs)
    results=target.get_statistics() | {"benchmark": benchmark, 
                                       "method": method, 
                                       "convergence": is_equivalent} | kwargs | {"total_time": endint_time - starting_time}

    
    writelog(logger, f"Finished — results: {results}", level="info")
    
    with open(os.path.join(kwargs.get("output_dir", "./results"), f"{benchmark}_{method}_{seed}_{alpha}_{time.time()}.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--method", choices=["quacq", "pquacq", "ml_quacq_e", "ml_quacq_n"])
    parser.add_argument("--benchmark", choices=["zebra", "jigsaw","rflap","random"],required=True)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--features", 
                    type=lambda s: s.split(",") if s and s.lower() != "none" else None, 
                    default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
   
    args = parser.parse_args()
    main(**args.__dict__)