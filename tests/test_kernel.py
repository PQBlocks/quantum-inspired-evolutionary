import numpy as np
import pytest
import sys
import os
from scipy import stats

# Ensure project root is in sys.path for direct pytest invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqblocks_qiea.qiea_kernel import _qiea_kernel, get_rotation_sign_numba, _compute_spin_energy_numba
from pqblocks_qiea.qiea_reference import _qiea_reference, get_rotation_sign, _compute_spin_energy

def test_get_rotation_sign_numba_matches_spec():
    """Test the Numba rotation sign function against the spec lookup table."""
    # Test case from spec: x_j=0, b_j=1, f(x) < f(b), alpha*beta > 0 -> s = +1
    assert get_rotation_sign_numba(0, 1, False, 0.7071, 0.7071) == +1
    
    # Test case from spec: x_j=0, b_j=1, f(x) < f(b), alpha*beta < 0 -> s = -1
    assert get_rotation_sign_numba(0, 1, False, -0.7071, 0.7071) == -1
    
    # Test case from spec: x_j=0, b_j=1, f(x) >= f(b), alpha*beta > 0 -> s = -1
    assert get_rotation_sign_numba(0, 1, True, 0.7071, 0.7071) == -1
    
    # Test case from spec: x_j=1, b_j=0, f(x) < f(b), alpha*beta > 0 -> s = -1
    assert get_rotation_sign_numba(1, 0, False, 0.7071, 0.7071) == -1
    
    # Test case from spec: x_j=1, b_j=0, f(x) >= f(b), alpha*beta > 0 -> s = +1
    assert get_rotation_sign_numba(1, 0, True, 0.7071, 0.7071) == +1
    
    # Test cases where no rotation occurs
    assert get_rotation_sign_numba(0, 0, False, 0.5, 0.5) == 0
    assert get_rotation_sign_numba(1, 1, True, 0.5, 0.5) == 0

def test_kernel_rotation_sign_matches_reference():
    """Test that the Numba rotation sign matches the reference Python version."""
    # Test all combinations of x, b, f_ge_fb
    test_cases = [
        (0, 0, False), (0, 0, True),
        (1, 1, False), (1, 1, True),
        (0, 1, False), (0, 1, True),
        (1, 0, False), (1, 0, True),
    ]
    
    # Test with different alpha, beta values
    alpha_beta_cases = [
        (0.7071, 0.7071),   # Both positive
        (-0.7071, 0.7071),  # alpha negative, beta positive
        (0.7071, -0.7071),  # alpha positive, beta negative
        (-0.7071, -0.7071), # Both negative
        (0.0, 1.0),         # alpha = 0
        (1.0, 0.0),         # beta = 0
        (0.0, 0.0),         # Both zero
    ]
    
    for x, b, f_ge_fb in test_cases:
        for alpha, beta in alpha_beta_cases:
            ref_result = get_rotation_sign(x, b, f_ge_fb, alpha, beta)
            numba_result = get_rotation_sign_numba(x, b, f_ge_fb, alpha, beta)
            assert ref_result == numba_result, f"Mismatch: x={x}, b={b}, f_ge_fb={f_ge_fb}, alpha={alpha}, beta={beta}"

def test_kernel_energy_matches_reference():
    """Test that the Numba energy computation matches the reference."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    test_states = [
        np.array([1, 0, 0, 0], dtype=np.int8),
        np.array([0, 1, 0, 1], dtype=np.int8),
        np.array([1, 1, 1, 1], dtype=np.int8),
        np.array([0, 0, 0, 0], dtype=np.int8),
        np.array([1, 0, 1, 0], dtype=np.int8),
    ]
    
    for x in test_states:
        ref_energy = _compute_spin_energy(x, h, J)
        numba_energy = _compute_spin_energy_numba(x, h, J)
        assert abs(ref_energy - numba_energy) < 1e-10, f"Energy mismatch for x={x}: ref={ref_energy}, numba={numba_energy}"

def test_kernel_matches_reference_on_4spin():
    """Test that the kernel matches the reference on the 4-spin worked example."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    num_generations = 10
    pop_size = 5
    theta_0 = 0.05 * np.pi
    seed = 42
    
    # Run both implementations
    ref_best_s, ref_best_e = _qiea_reference(h, J, num_generations, pop_size, theta_0, seed)
    kernel_best_s, kernel_best_e = _qiea_kernel(h, J, num_generations, pop_size, theta_0, seed)
    
    # They should produce the exact same results with the same seed
    # (since both use np.random.seed() for reproducibility)
    np.testing.assert_array_equal(ref_best_s, kernel_best_s)
    assert abs(ref_best_e - kernel_best_e) < 1e-10

def test_kernel_ks_test_equivalence():
    """
    Statistical equivalence test using two-sample Kolmogorov-Smirnov test.
    
    Over 200 independently seeded runs on a 50-spin instance, extract the final 
    best energies from the reference oracle and the optimized kernel. 
    A two-sample KS test must yield p > 0.01, validating statistical indistinguishability.
    """
    # Create a 50-spin random instance
    N = 50
    np.random.seed(12345)
    h = np.random.uniform(-1.0, 1.0, N)
    # Create symmetric J with zero diagonal
    J = np.random.uniform(-1.0, 1.0, (N, N))
    J = (J + J.T) / 2.0  # Make symmetric
    for i in range(N):
        J[i, i] = 0.0  # Zero diagonal
    
    num_generations = 20
    pop_size = 10
    theta_0 = 0.05 * np.pi
    
    # Run 200 independent seeds per spec Section 6 & 8 (or 50 if JIT disabled for speed)
    n_runs = 50 if os.environ.get("NUMBA_DISABLE_JIT", "0") == "1" else 200
    ref_energies = np.zeros(n_runs)
    kernel_energies = np.zeros(n_runs)
    
    for i in range(n_runs):
        seed = i * 100 + 42  # Different seed for each run
        
        # Reference
        _, ref_e = _qiea_reference(h, J, num_generations, pop_size, theta_0, seed)
        ref_energies[i] = ref_e
        
        # Kernel
        _, kernel_e = _qiea_kernel(h, J, num_generations, pop_size, theta_0, seed)
        kernel_energies[i] = kernel_e
    
    # Perform two-sample KS test
    ks_stat, p_value = stats.ks_2samp(ref_energies, kernel_energies)
    
    # The KS test should yield p > 0.01 for statistical equivalence
    assert p_value > 0.01, f"KS test failed: p={p_value:.6f} <= 0.01. The distributions are statistically different."


def test_kernel_basic_execution():
    """Basic test that the kernel runs and returns reasonable results."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    best_s, best_e = _qiea_kernel(h, J, 10, 5, 0.05 * np.pi, seed=42)
    
    assert best_s is not None
    assert isinstance(best_e, (int, float))
    assert len(best_s) == len(h)
    assert np.all(np.isin(best_s, [-1, 1]))
    assert np.isfinite(best_e)


def test_kernel_execution_seed_none():
    """Test kernel execution when seed is None."""
    h = np.array([1.0, -1.0])
    J = np.array([[0.0, 1.0], [1.0, 0.0]])
    best_s, best_e = _qiea_kernel(h, J, 5, 4, 0.05 * np.pi, seed=None)
    assert best_s is not None
    assert len(best_s) == 2
    assert np.all(np.isin(best_s, [-1, 1]))
    assert np.isfinite(best_e)


def test_kernel_single_spin():
    """Test kernel execution on single spin N=1."""
    h = np.array([1.5])
    J = np.zeros((1, 1), dtype=np.float64)
    best_s, best_e = _qiea_kernel(h, J, 10, 4, 0.05 * np.pi, seed=42)
    assert len(best_s) == 1
    assert best_s[0] == -1  # Minimizes 1.5 * s -> s = -1, E = -1.5
    assert abs(best_e - (-1.5)) < 1e-10


def test_kernel_all_zero_model():
    """Test kernel execution on all-zero model."""
    h = np.zeros(3, dtype=np.float64)
    J = np.zeros((3, 3), dtype=np.float64)
    best_s, best_e = _qiea_kernel(h, J, 5, 4, 0.05 * np.pi, seed=42)
    assert len(best_s) == 3
    assert abs(best_e - 0.0) < 1e-10


def test_get_rotation_sign_numba_fallback():
    """Test fallback return 0 branch for invalid bit values."""
    assert get_rotation_sign_numba(2, 3, False, 0.5, 0.5) == 0