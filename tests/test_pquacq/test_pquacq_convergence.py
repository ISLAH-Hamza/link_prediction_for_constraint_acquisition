import sys
import pytest

from benchmarks import zebra, jigsaw
from core.base import TargetNetwork
from acquisition.baselines.pquacq import pquacq, adamic_adar
from core.solve import equivalent


# We run each convergence test once (parametrize range can be increased to reduce flakiness)

@pytest.mark.parametrize("iteration", range(1))
def test_zebra_convergence(iteration):
    zebra_puzzle = zebra.Model()
    variables = zebra_puzzle.variables
    B = zebra_puzzle.bais
    target = TargetNetwork(zebra_puzzle.constraints)
    L = pquacq(alpha=3, score=adamic_adar, B=B, target_network=target, variables=variables)
    assert equivalent(target.constraints, L, variables)


@pytest.mark.parametrize("iteration", range(1))
def test_jigsaw_convergence(iteration):
    jigsaw_puzzle = jigsaw.Model()
    variables = jigsaw_puzzle.variables
    B = jigsaw_puzzle.bais
    target = TargetNetwork(jigsaw_puzzle.constraints)
    L = pquacq(alpha=3, score=adamic_adar, B=B, target_network=target, variables=variables)
    assert equivalent(target.constraints, L, variables)


if __name__ == "__main__":
    print("WARNING: These tests may take significant time to complete.")
    response = input("Do you want to continue? (y/n): ").strip().lower()
    if response == "y":
        pytest.main([__file__])
    else:
        print("Tests aborted by user.")
        sys.exit(0)