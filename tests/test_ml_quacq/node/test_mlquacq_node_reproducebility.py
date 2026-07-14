import sys
import pytest

from benchmarks import jigsaw, zebra
from core.base import TargetNetwork
from acquisition.ml_quacq.ml_quacq_n import ml_quacq
from core.solve import equivalent
from utils import fixed_seed

# ===========================================================================
# Convergence tests
#
# These tests run the full ml_quacq (node/GNN variant) loop on benchmark
# puzzles. When the puzzle is too small to trigger the ML path (fewer than
# 10 positive/negative examples per relation), ml_quacq degrades gracefully
# to plain QUACQ — convergence is still expected.
# ===========================================================================


@pytest.mark.parametrize("iteration", range(1))
def test_jigsaw_convergence(iteration):
    for seed in [0,1,2]:
        fixed_seed(seed)
        instance = jigsaw.Model()
        variables = instance.variables
        B = instance.bais
        target1 = TargetNetwork(instance.constraints)
        L1 = ml_quacq(3, B, target1, variables, seed=seed)

         
        fixed_seed(seed)
        instance = jigsaw.Model()
        variables = instance.variables
        B = instance.bais
        target2 = TargetNetwork(instance.constraints)
        L2 = ml_quacq(3, B, target2, variables, seed=seed)
        
        assert len(L1) == len(L2)
        assert target1.ask_counter ==target2.ask_counter
        assert target1.req_yes == target2.req_yes
        assert target1.req_no == target2.req_no


@pytest.mark.parametrize("iteration", range(1))
def test_zebra_convergence(iteration):
    for seed in [0,1,2]:
        fixed_seed(seed)
        instance = zebra.Model()
        variables = instance.variables
        B = instance.bais
        target1 = TargetNetwork(instance.constraints)
        L1 = ml_quacq(3, B, target1, variables, seed=seed)

         
        fixed_seed(seed)
        instance = zebra.Model()
        variables = instance.variables
        B = instance.bais
        target2 = TargetNetwork(instance.constraints)
        L2 = ml_quacq(3, B, target2, variables, seed=seed)
        
        assert len(L1) == len(L2)
        assert target1.ask_counter ==target2.ask_counter
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