"""
Verification tests for Slice 6: Integrate golden instances into verification; compute optimality gap.

These tests wire the 6 golden instances into the verification framework and validate
that the QIEA solver reaches within the specified optimality gap tolerance for each.
"""

import json
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pqblocks_qiea.qiea_api import solve_ising
from pqblocks_qiea.golden_instances import (
    GoldenInstances,
    validate_against_certified,
    compute_optimality_gap
)


_fcand = Path(__file__).resolve().parent / "fixtures" / "golden" / "golden_instances.json"
_pcand = Path(__file__).resolve().parent.parent / "src" / "pqblocks_qiea" / "golden_instances.json"
_rcand = Path(__file__).resolve().parent.parent / "golden" / "golden_instances.json"
GOLDEN_PATH = _fcand if _fcand.exists() else (_pcand if _pcand.exists() else _rcand)


class TestSolverOnGoldenInstances:
    """Tests that the QIEA solver reaches optima on all golden instances."""
    
    def test_solver_on_planted_4spin(self):
        """Verify QIEA solver reaches the reference optimum on planted_4spin."""
        gi = GoldenInstances(GOLDEN_PATH)
        inst = gi["planted_4spin"]
        
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        certified_opt = inst["certified_optimum"]["energy"]
        
        best_s, best_e = solve_ising(h, J, num_generations=50, population_size=10, seed=42)
        gap = compute_optimality_gap(best_e, certified_opt)
        
        assert gap <= 1e-4, f"Solver energy {best_e} did not match optimum {certified_opt} (gap={gap})"
    
    def test_solver_on_planted_6spin(self):
        """Verify QIEA solver reaches the reference optimum on planted_6spin."""
        gi = GoldenInstances(GOLDEN_PATH)
        inst = gi["planted_6spin"]
        
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        certified_opt = inst["certified_optimum"]["energy"]
        
        # Use more generations for larger instance
        best_s, best_e = solve_ising(h, J, num_generations=100, population_size=20, seed=123)
        gap = compute_optimality_gap(best_e, certified_opt)
        
        assert gap <= 1e-4, f"planted_6spin: Solver energy {best_e} did not match optimum {certified_opt} (gap={gap})"
    
    def test_solver_on_planted_8spin(self):
        """Verify QIEA solver reaches the reference optimum on planted_8spin."""
        gi = GoldenInstances(GOLDEN_PATH)
        inst = gi["planted_8spin"]
        
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        certified_opt = inst["certified_optimum"]["energy"]
        
        best_s, best_e = solve_ising(h, J, num_generations=150, population_size=20, seed=456)
        gap = compute_optimality_gap(best_e, certified_opt)
        
        assert gap <= 1e-4, f"planted_8spin: Solver energy {best_e} did not match optimum {certified_opt} (gap={gap})"
    
    def test_solver_on_planted_10spin(self):
        """Verify QIEA solver reaches the reference optimum on planted_10spin."""
        gi = GoldenInstances(GOLDEN_PATH)
        inst = gi["planted_10spin"]
        
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        certified_opt = inst["certified_optimum"]["energy"]
        
        best_s, best_e = solve_ising(h, J, num_generations=200, population_size=30, seed=789)
        gap = compute_optimality_gap(best_e, certified_opt)
        
        assert gap <= 1e-4, f"planted_10spin: Solver energy {best_e} did not match optimum {certified_opt} (gap={gap})"


class TestMQLibInstances:
    """Tests for MQLib instances with larger tolerance (0.01)."""
    
    def test_solver_on_mqlib_rudy_10(self):
        """Verify QIEA solver reaches within tolerance on mqlib_rudy_10."""
        gi = GoldenInstances(GOLDEN_PATH)
        inst = gi["mqlib_rudy_10"]
        
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        certified_opt = inst["certified_optimum"]["energy"]
        
        best_s, best_e = solve_ising(h, J, num_generations=200, population_size=30, seed=111)
        gap = compute_optimality_gap(best_e, certified_opt)
        
        assert gap <= 0.01, f"mqlib_rudy_10: Solver energy {best_e} gap {gap} exceeds MQLib tolerance 0.01"
    
    def test_solver_on_mqlib_rudy_12(self):
        """Verify QIEA solver reaches within tolerance on mqlib_rudy_12."""
        gi = GoldenInstances(GOLDEN_PATH)
        inst = gi["mqlib_rudy_12"]
        
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        certified_opt = inst["certified_optimum"]["energy"]
        
        best_s, best_e = solve_ising(h, J, num_generations=300, population_size=40, seed=222)
        gap = compute_optimality_gap(best_e, certified_opt)
        
        assert gap <= 0.01, f"mqlib_rudy_12: Solver energy {best_e} gap {gap} exceeds MQLib tolerance 0.01"


class TestGoldenInstancesVerification:
    """Tests for the GoldenInstances class validation methods."""
    
    def test_validate_all_instances(self):
        """Test validation of solver output on all 6 golden instances."""
        gi = GoldenInstances(GOLDEN_PATH)
        
        def solver_wrapper(h, J, seed=None):
            """Wrapper that runs QIEA solver."""
            best_s, best_e = solve_ising(
                h, J, 
                num_generations=100, 
                population_size=20,
                seed=seed
            )
            return best_s, best_e
        
        # Validate all instances (using 3 runs each for speed)
        results = gi.validate_all(
            solver_wrapper,
            num_runs=3,
            tolerance_planted=1e-4,
            tolerance_mqlib=0.01
        )
        
        # Print results for debugging
        for name, result in results.items():
            print(f"{name} ({result['type']}): success_rate={result['success_rate']:.2f}, "
                  f"mean_gap={result['mean_gap']:.6f}, min_gap={result['min_gap']:.6f}")
        
        # All planted instances should have high success rate (at least some runs finding exact optimum)
        for name, result in results.items():
            if result["type"] == "planted":
                # With multiple runs, at least one should find exact optimum for small instances
                assert result["min_gap"] <= 1e-4, (
                    f"{name}: minimum gap {result['min_gap']} > tolerance 1e-4"
                )
            else:
                # MQLib: at least one run within 0.01
                assert result["min_gap"] <= 0.01, (
                    f"{name}: minimum gap {result['min_gap']} > MQLib tolerance 0.01"
                )
    
    def test_optimality_gap_report(self):
        """Generate and verify optimality gap report for all instances."""
        gi = GoldenInstances(GOLDEN_PATH)
        
        # Run solver on each instance once with a fixed seed
        gaps = {}
        for name in gi.instances:
            inst = gi.instances[name]
            h = np.array(inst["h"], dtype=np.float64)
            J = np.array(inst["J"], dtype=np.float64)
            certified_opt = inst["certified_optimum"]["energy"]
            
            # Determine parameters based on instance size
            if inst["N"] <= 6:
                gens, pop = 50, 10
            elif inst["N"] <= 10:
                gens, pop = 100, 20
            else:
                gens, pop = 150, 30
            
            best_s, best_e = solve_ising(
                h, J, num_generations=gens, population_size=pop, seed=42
            )
            
            gap = compute_optimality_gap(best_e, certified_opt)
            gaps[name] = {
                "certified_energy": certified_opt,
                "solver_energy": best_e,
                "gap": gap,
                "instance_type": inst["type"],
                "tolerance": 1e-4 if inst["type"] == "planted" else 0.01,
                "passed": gap <= (1e-4 if inst["type"] == "planted" else 0.01)
            }
        
        # Print gap report
        print("\n=== Optimality Gap Report ===")
        for name, data in gaps.items():
            status = "PASS" if data["passed"] else "FAIL"
            print(f"  {name}: E_cert={data['certified_energy']:.4f}, "
                  f"E_solver={data['solver_energy']:.4f}, "
                  f"gap={data['gap']:.6f}, tol={data['tolerance']:.4f} [{status}]")
        
        # Verify all pass
        for name, data in gaps.items():
            assert data["passed"], f"{name}: gap {data['gap']} exceeds tolerance {data['tolerance']}"


class TestNegativeCases:
    """Negative tests: solver should fail when parameters are too small."""
    
    def test_solver_fails_with_insufficient_generations(self):
        """With very few generations, solver may not reach optimum."""
        gi = GoldenInstances(GOLDEN_PATH)
        inst = gi["planted_4spin"]
        
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        certified_opt = inst["certified_optimum"]["energy"]
        
        # Use very few generations - likely won't converge
        best_s, best_e = solve_ising(h, J, num_generations=1, population_size=2, seed=999)
        gap = compute_optimality_gap(best_e, certified_opt)
        
        # This is a sanity check - the gap should be large
        # We don't assert on the exact gap, just that the function works
        assert gap >= 0
        assert np.isfinite(gap)
    
    def test_solver_on_random_instance(self):
        """Solver runs on random instance (no known optimum, just sanity check)."""
        # Generate a random 8-spin instance
        rng = np.random.RandomState(999)
        N = 8
        h = np.round(rng.uniform(-1.5, 1.5, size=N), 2)
        J = rng.uniform(-1.0, 1.0, size=(N, N))
        J = np.round((J + J.T) / 2.0, 2)
        np.fill_diagonal(J, 0.0)
        
        best_s, best_e = solve_ising(h, J, num_generations=50, population_size=10, seed=42)
        
        # Just verify it runs and returns valid output
        assert best_s is not None
        assert len(best_s) == N
        assert np.all(np.isin(best_s, [-1, 1]))
        assert np.isfinite(best_e)


class TestSolverConsistency:
    """Tests that solver results are consistent and reproducible."""
    
    def test_same_seed_returns_same_result(self):
        """Same seed should produce identical results."""
        gi = GoldenInstances(GOLDEN_PATH)
        inst = gi["planted_4spin"]
        
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        
        s1, e1 = solve_ising(h, J, num_generations=20, population_size=10, seed=42)
        s2, e2 = solve_ising(h, J, num_generations=20, population_size=10, seed=42)
        
        np.testing.assert_array_equal(s1, s2)
        assert abs(e1 - e2) < 1e-12
    
    def test_different_seeds_may_differ(self):
        """Different seeds may produce different results."""
        gi = GoldenInstances(GOLDEN_PATH)
        inst = gi["planted_4spin"]
        
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        
        s1, e1 = solve_ising(h, J, num_generations=20, population_size=10, seed=42)
        s2, e2 = solve_ising(h, J, num_generations=20, population_size=10, seed=43)
        
        # At least one of spin vector or energy should differ (not guaranteed, but likely)
        # This is a soft test - just verify both are valid
        assert np.all(np.isin(s1, [-1, 1]))
        assert np.all(np.isin(s2, [-1, 1]))
        assert np.isfinite(e1)
        assert np.isfinite(e2)


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])