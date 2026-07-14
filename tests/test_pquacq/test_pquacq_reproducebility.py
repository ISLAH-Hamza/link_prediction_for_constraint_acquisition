import sys
import pytest

from benchmarks import zebra, jigsaw
from core.base import TargetNetwork
from acquisition.baselines.pquacq import pquacq, adamic_adar
from core.solve import equivalent
from utils import fixed_seed


@pytest.mark.parametrize("iteration", range(1))
def test_zebra_convergence(iteration):
    for seed in [42, 67, 123]:
        fixed_seed(seed)
        zebra_puzzle = zebra.Model()
        variables = zebra_puzzle.variables
        B = zebra_puzzle.bais
        target1 = TargetNetwork(zebra_puzzle.constraints)
        L1 = pquacq(alpha=3, score=adamic_adar, B=B, target_network=target1, variables=variables, seed = seed)
    
        fixed_seed(seed)
        zebra_puzzle = zebra.Model()
        variables = zebra_puzzle.variables
        B = zebra_puzzle.bais
        target2 = TargetNetwork(zebra_puzzle.constraints)
        L2 = pquacq(alpha=3, score=adamic_adar, B=B, target_network=target2, variables=variables, seed = seed)

        assert(len(L1) == len(L2))
        assert target1.ask_counter == target2.ask_counter
        assert target1.req_yes == target2.req_yes
        assert target1.req_no == target2.req_no


@pytest.mark.parametrize("iteration", range(1))
def test_jigsaw_convergence(iteration):
    for seed in [5, 10, 442]:
        fixed_seed(seed)
        jigsaw_puzzle = jigsaw.Model()
        variables = jigsaw_puzzle.variables
        B = jigsaw_puzzle.bais
        target1 = TargetNetwork(jigsaw_puzzle.constraints)
        L1 = pquacq(alpha=2, score=adamic_adar, B=B, target_network=target1, variables=variables, seed=seed)
    
        fixed_seed(seed)
        jigsaw_puzzle = jigsaw.Model()
        variables = jigsaw_puzzle.variables
        B = jigsaw_puzzle.bais
        target2 = TargetNetwork(jigsaw_puzzle.constraints)
        L2 = pquacq(alpha=2, score=adamic_adar, B=B, target_network=target2, variables=variables, seed=seed)

        assert(len(L1) == len(L2))
        assert target1.ask_counter == target2.ask_counter
        assert target1.req_yes == target2.req_yes
        assert target1.req_no == target2.req_no
   

if __name__ == "__main__":
    print("WARNING: These tests may take significant time to complete.")
    response = input("Do you want to continue? (y/n): ").strip().lower()
    if response == "y":
        pytest.main([__file__])
    else:
        print("Tests aborted by user.")
        sys.exit(0)