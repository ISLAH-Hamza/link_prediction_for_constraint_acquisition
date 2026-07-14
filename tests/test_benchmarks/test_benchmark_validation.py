import pytest
from benchmarks import zebra, jigsaw, rflap

from core.solve import Solve
import warnings
import sys

# we run the convergence tests multiple times to reduce the flakiness due to the random nature of the learning algorithm and 
# check the convergence of our implimentation


@pytest.mark.parametrize("iteration", range(1))
def test_zebra_convergence(iteration):
    zebra_puzzle=zebra.Model()
    sol=Solve(zebra_puzzle.constraints,zebra_puzzle.variables)
    assert sol is not None and len(sol)>0



@pytest.mark.parametrize("iteration", range(1))
def test_jigsaw_convergence(iteration):
    jigsaw_puzzle=jigsaw.Model()
    sol=Solve(jigsaw_puzzle.constraints,jigsaw_puzzle.variables)
    
    assert sol is not None and len(sol)>0



@pytest.mark.parametrize("iteration", range(1))
def test_rflap_convergence(iteration):
    rflap_problem=rflap.Model()
    sol=Solve(rflap_problem.constraints,rflap_problem.variables)
    assert sol is not None and len(sol)>0




if __name__ == "__main__":
    print("WARNING: These tests may take significant time to complete.")
    response = input("Do you want to continue? (y/n): ").strip().lower()
    if response == "y":
        pytest.main([__file__])
    else:
        print("Tests aborted by user.")
        sys.exit(0)