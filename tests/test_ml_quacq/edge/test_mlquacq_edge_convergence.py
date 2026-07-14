import sys
import pytest

from benchmarks import jigsaw, zebra
from core.base import TargetNetwork
from acquisition.ml_quacq.ml_quacq_e import ml_quacq, fit_link_predictor
from core.solve import equivalent


# ===========================================================================
# Convergence tests
#
# These tests run the full ml_quacq loop on benchmark puzzles.
# When the puzzle is too small to trigger the ML path (fewer than 10
# positive/negative examples per relation), ml_quacq degrades gracefully
# to plain QUACQ — convergence is still expected.
# ===========================================================================


@pytest.mark.parametrize("iteration", range(1))
def test_jigsaw_convergence(iteration):
    instance = jigsaw.Model()
    variables = instance.variables
    B = instance.bais
    target = TargetNetwork(instance.constraints)

    L = ml_quacq(3, B, target, variables)

    assert equivalent(target.constraints, L, variables)


@pytest.mark.parametrize("iteration", range(1))
def test_zebra_convergence(iteration):
    instance = zebra.Model()
    variables = instance.variables
    B = instance.bais
    target = TargetNetwork(instance.constraints)

    L = ml_quacq(3, B, target, variables)

    assert equivalent(target.constraints, L, variables)


if __name__ == "__main__":
    print("WARNING: These tests may take significant time to complete.")
    response = input("Do you want to continue? (y/n): ").strip().lower()
    if response == "y":
        pytest.main([__file__])
    else:
        print("Tests aborted by user.")
        sys.exit(0)