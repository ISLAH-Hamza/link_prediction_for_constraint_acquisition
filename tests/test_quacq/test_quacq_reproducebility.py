import pytest
from benchmarks import zebra, jigsaw
from core.base import  TargetNetwork
from acquisition.baselines.quacq import QuAcq
from core.solve import equivalent
import warnings
import sys
from utils import fixed_seed

# we run the convergence tests multiple times to reduce the flakiness due to the random nature of the learning algorithm and 
# check the convergence of our implimentation

@pytest.mark.parametrize("iteration", range(1))
def test_zebra_convergence(iteration):
    for seed in [0, 1, 2]:
        fixed_seed(seed)
        zebra_puzzle=zebra.Model()
        variables=zebra_puzzle.variables
        B=zebra_puzzle.bais
        target1=TargetNetwork(zebra_puzzle.constraints)
        L1=QuAcq(B,target1,variables, seed=seed)
        
        fixed_seed(seed)
        zebra_puzzle=zebra.Model()
        variables=zebra_puzzle.variables
        B=zebra_puzzle.bais
        target2=TargetNetwork(zebra_puzzle.constraints)
        L2=QuAcq(B,target2,variables, seed=seed)

        assert(len(L1) == len(L2))
        assert(target1.ask_counter == target2.ask_counter)
    

@pytest.mark.parametrize("iteration", range(1))
def test_jigsaw_convergence(iteration):
    for seed in [5, 10, 442]:
        fixed_seed(seed)
        jigsaw_puzzle=jigsaw.Model()
        variables=jigsaw_puzzle.variables
        B=jigsaw_puzzle.bais
        target1=TargetNetwork(jigsaw_puzzle.constraints)
        L1=QuAcq(B,target1,variables, seed=seed)

        fixed_seed(seed)
        jigsaw_puzzle=jigsaw.Model()
        variables=jigsaw_puzzle.variables
        B=jigsaw_puzzle.bais
        target2=TargetNetwork(jigsaw_puzzle.constraints)
        L2=QuAcq(B,target2,variables, seed=seed)

        assert(len(L1) == len(L2))
        assert(target1.ask_counter == target2.ask_counter)


if __name__ == "__main__":
    print("WARNING: These tests may take significant time to complete.")
    response = input("Do you want to continue? (y/n): ").strip().lower()
    if response == "y":
        pytest.main([__file__])
    else:
        print("Tests aborted by user.")
        sys.exit(0)