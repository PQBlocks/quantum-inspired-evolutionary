import numpy as np
import pytest
import sys
import os

# Add the project root to the path
_fix_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
if os.path.isdir(_fix_dir) and _fix_dir not in sys.path:
    sys.path.insert(0, _fix_dir)

from bench.baselines.simulated_annealing import simulated_annealing
from bench.baselines.tabu_search import tabu_search
from bench.utils.tts_measurement import measure_tts, compute_tts_statistics, measure_solution_quality, kolmogorov_smirnov_test


def test_simulated_annealing_basic():
    """Test that simulated annealing runs without error on a simple problem."""
    # Simple 2-spin ferromagnetic problem
    h = np.array([0.0, 0.0])
    J = np.array([[-1.0, 1.0], [1.0, -1.0]])
    
    best_spin, best_energy = simulated_annealing(h, J, num_sweeps=10, seed=42)
    
    assert isinstance(best_spin, np.ndarray)
    assert best_spin.shape == (2,)
    assert np.all(np.isin(best_spin, [-1, 1]))
    assert isinstance(best_energy, (int, float))
    assert np.isfinite(best_energy)


def test_tabu_search_basic():
    """Test that tabu search runs without error on a simple problem."""
    # Simple 2-spin ferromagnetic problem
    h = np.array([0.0, 0.0])
    J = np.array([[-1.0, 1.0], [1.0, -1.0]])
    
    best_spin, best_energy = tabu_search(h, J, num_iterations=10, seed=42)
    
    assert isinstance(best_spin, np.ndarray)
    assert best_spin.shape == (2,)
    assert np.all(np.isin(best_spin, [-1, 1]))
    assert isinstance(best_energy, (int, float))
    assert np.isfinite(best_energy)


def test_measure_tts():
    """Test TTS measurement functionality."""
    # Simple problem where we know the optimum
    h = np.array([0.0, 0.0])
    J = np.array([[-1.0, 1.0], [1.0, -1.0]])  # Optimum energy = -2.0
    
    def mock_solver(h, J, seed=None):
        """Mock solver that always finds the optimum."""
        if seed is not None:
            np.random.seed(seed)
        # Return the known optimum: [1, 1] or [-1, -1] -> spins [1, 1] or [-1, -1]
        # Energy = -2.0 for both
        best_spin = np.array([1, 1])  # or [-1, -1]
        best_energy = -2.0
        return best_spin, best_energy
    
    tts_list, num_success = measure_tts(
        mock_solver, h, J, target_energy=-2.0, tolerance=1e-6, num_seeds=5
    )
    
    assert isinstance(tts_list, list)
    assert len(tts_list) == 5  # All should succeed
    assert num_success == 5
    assert all(t >= 0 for t in tts_list)  # TTS should be non-negative


def test_compute_tts_statistics():
    """Test TTS statistics computation."""
    # Test with normal values
    tts_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = compute_tts_statistics(tts_list)
    
    assert stats['median_tts'] == 3.0
    assert stats['mean_tts'] == 3.0
    assert stats['num_samples'] == 5
    assert stats['percentile_25'] == 2.0
    assert stats['percentile_75'] == 4.0
    
    # Test with empty list
    empty_stats = compute_tts_statistics([])
    assert empty_stats['median_tts'] == float('inf')
    assert empty_stats['num_samples'] == 0


def test_measure_solution_quality():
    """Test solution quality measurement."""
    # Simple problem where we know the optimum
    h = np.array([0.0, 0.0])
    J = np.array([[-1.0, 1.0], [1.0, -1.0]])  # Optimum energy = -2.0
    
    def mock_solver_perfect(h, J, seed=None):
        """Mock solver that always finds the optimum."""
        if seed is not None:
            np.random.seed(seed)
        best_spin = np.array([1, 1])
        best_energy = -2.0
        return best_spin, best_energy
    
    def mock_solver_half(h, J, seed=None):
        """Mock solver that finds optimum half the time."""
        if seed is not None:
            np.random.seed(seed)
        if seed % 2 == 0:
            best_spin = np.array([1, 1])
            best_energy = -2.0
        else:
            best_spin = np.array([1, -1])  # Higher energy state
            best_energy = 0.0
        return best_spin, best_energy
    
    # Test perfect solver
    success_rate, mean_gap, gaps = measure_solution_quality(
        mock_solver_perfect, h, J, target_energy=-2.0, tolerance=1e-6, num_seeds=10
    )
    
    assert success_rate == 1.0
    assert mean_gap == 0.0
    assert all(g == 0.0 for g in gaps)
    
    # Test 50% solver
    success_rate, mean_gap, gaps = measure_solution_quality(
        mock_solver_half, h, J, target_energy=-2.0, tolerance=1e-6, num_seeds=10
    )
    
    assert success_rate == 0.5
    assert mean_gap == 1.0  # Average of 0.0 and 2.0
    assert len(gaps) == 10


def test_kolmogorov_smirnov_test():
    """Test KS test functionality."""
    # Test with identical distributions
    sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    sample2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    ks_stat, p_value, passes = kolmogorov_smirnov_test(sample1, sample2, alpha=0.01)
    
    assert isinstance(ks_stat, float)
    assert isinstance(p_value, float)
    # Accept both Python bool and numpy bool_ as they are compatible
    assert isinstance(passes, (bool, np.bool_))
    assert passes == True  # Identical distributions should pass
    
    # Test with very different distributions
    sample3 = [1.0, 1.1, 1.2, 1.3, 1.4]
    sample4 = [5.0, 5.1, 5.2, 5.3, 5.4]
    
    ks_stat2, p_value2, passes2 = kolmogorov_smirnov_test(sample3, sample4, alpha=0.01)
    
    assert passes2 == False  # Very different distributions should fail
    
    # Test with insufficient data
    ks_stat3, p_value3, passes3 = kolmogorov_smirnov_test([1.0], [2.0], alpha=0.01)
    
    assert passes3 == True  # Should pass when not enough data