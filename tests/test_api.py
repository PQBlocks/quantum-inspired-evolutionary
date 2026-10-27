import numpy as np
import pytest
import sys
import os

# Ensure project root is in sys.path for direct pytest invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqblocks_qiea.qiea_api import solve_ising
from pqblocks_qiea.qiea_kernel import _qiea_kernel

def test_solve_ising_basic():
    """Test basic solve_ising functionality."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    best_s, best_e = solve_ising(h, J, num_generations=10, population_size=5, seed=42)
    
    assert best_s is not None
    assert isinstance(best_e, (int, float))
    assert len(best_s) == len(h)
    assert np.all(np.isin(best_s, [-1, 1]))
    assert np.isfinite(best_e)

def test_solve_ising_empty_model():
    """Test that empty model (N=0) raises ValueError."""
    h = np.array([])
    J = np.array([]).reshape(0, 0)
    
    with pytest.raises(ValueError, match="Empty model: N=0 is not supported"):
        solve_ising(h, J, num_generations=10, population_size=5, seed=42)

def test_solve_ising_dimension_mismatch():
    """Test that dimension mismatch raises ValueError."""
    h = np.array([1.0, -1.0, 0.5])  # N=3
    J = np.array([  # 4x4 matrix
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    with pytest.raises(ValueError, match="Dimension mismatch between h and J"):
        solve_ising(h, J, num_generations=10, population_size=5, seed=42)

def test_solve_ising_non_symmetric_J():
    """Test that non-symmetric J raises ValueError."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, -1.0, 0.0],  # J[1,0] != J[0,1]
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    with pytest.raises(ValueError, match="Coupling matrix J must be symmetric"):
        solve_ising(h, J, num_generations=10, population_size=5, seed=42)

def test_solve_ising_non_zero_diagonal():
    """Test that non-zero diagonal raises ValueError."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [1.0, 1.0, 0.0, 0.0],  # Non-zero diagonal
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    with pytest.raises(ValueError, match="Coupling matrix J must have a zero diagonal"):
        solve_ising(h, J, num_generations=10, population_size=5, seed=42)

def test_solve_ising_nan_in_h():
    """Test that NaN in h raises ValueError."""
    h = np.array([1.0, np.nan, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    with pytest.raises(ValueError, match="h and J must contain only finite numbers"):
        solve_ising(h, J, num_generations=10, population_size=5, seed=42)

def test_solve_ising_inf_in_J():
    """Test that Inf in J raises ValueError."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, np.inf],  # Inf in J
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    with pytest.raises(ValueError, match="h and J must contain only finite numbers"):
        solve_ising(h, J, num_generations=10, population_size=5, seed=42)

def test_solve_ising_invalid_num_generations():
    """Test that invalid num_generations raises ValueError."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    with pytest.raises(ValueError, match="Number of generations must be >= 1"):
        solve_ising(h, J, num_generations=0, population_size=5, seed=42)
    
    with pytest.raises(ValueError, match="Number of generations must be >= 1"):
        solve_ising(h, J, num_generations=-1, population_size=5, seed=42)

def test_solve_ising_invalid_population_size():
    """Test that invalid population_size raises ValueError."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    with pytest.raises(ValueError, match="Population size must be >= 1"):
        solve_ising(h, J, num_generations=10, population_size=0, seed=42)
    
    with pytest.raises(ValueError, match="Population size must be >= 1"):
        solve_ising(h, J, num_generations=10, population_size=-1, seed=42)

def test_solve_ising_invalid_theta_0():
    """Test that invalid theta_0 raises ValueError."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    with pytest.raises(ValueError, match="Theta_0 must be > 0"):
        solve_ising(h, J, num_generations=10, population_size=5, theta_0=0.0, seed=42)
    
    with pytest.raises(ValueError, match="Theta_0 must be > 0"):
        solve_ising(h, J, num_generations=10, population_size=5, theta_0=-0.1, seed=42)
    
    with pytest.raises(ValueError, match="Theta_0 must be <= pi/2"):
        solve_ising(h, J, num_generations=10, population_size=5, theta_0=np.pi, seed=42)
    
    with pytest.raises(ValueError, match="Theta_0 must be <= pi/2"):
        solve_ising(h, J, num_generations=10, population_size=5, theta_0=2.0, seed=42)

def test_solve_ising_auto_population_size():
    """Test that population_size is auto-derived correctly."""
    # For N=4, auto pop_size = max(10, min(100, 2*4)) = max(10, min(100, 8)) = 10
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # Should not raise an error and should use auto-derived pop_size=10
    best_s, best_e = solve_ising(h, J, num_generations=10, seed=42)
    
    assert best_s is not None
    assert len(best_s) == 4
    assert np.all(np.isin(best_s, [-1, 1]))

def test_solve_ising_single_spin():
    """Test single spin (N=1) case - should proceed normally."""
    h = np.array([1.0])
    J = np.array([[0.0]])  # Zero diagonal
    
    # Should not raise an error
    best_s, best_e = solve_ising(h, J, num_generations=10, population_size=5, seed=42)
    
    assert best_s is not None
    assert len(best_s) == 1
    assert best_s[0] in [-1, 1]
    assert np.isfinite(best_e)

def test_solve_ising_all_zero_model():
    """Test all-zero model - should proceed normally."""
    h = np.array([0.0, 0.0, 0.0])
    J = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    # Should not raise an error
    best_s, best_e = solve_ising(h, J, num_generations=10, population_size=5, seed=42)
    
    assert best_s is not None
    assert len(best_s) == 3
    assert np.all(np.isin(best_s, [-1, 1]))
    assert np.isfinite(best_e)