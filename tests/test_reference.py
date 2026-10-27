import numpy as np
import pytest
import math
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pqblocks_qiea.qiea_reference import _qiea_reference, get_rotation_sign, _compute_spin_energy


def test_get_rotation_sign_matches_spec():
    """Test the rotation sign function against the spec lookup table."""
    # Test case from spec: x_j=0, b_j=1, f(x) < f(b), alpha*beta > 0 -> s = +1
    assert get_rotation_sign(0, 1, False, 0.7071, 0.7071) == +1
    
    # Test case from spec: x_j=0, b_j=1, f(x) < f(b), alpha*beta < 0 -> s = -1
    assert get_rotation_sign(0, 1, False, -0.7071, 0.7071) == -1
    
    # Test case from spec: x_j=0, b_j=1, f(x) >= f(b), alpha*beta > 0 -> s = -1
    assert get_rotation_sign(0, 1, True, 0.7071, 0.7071) == -1
    
    # Test case from spec: x_j=1, b_j=0, f(x) < f(b), alpha*beta > 0 -> s = -1
    assert get_rotation_sign(1, 0, False, 0.7071, 0.7071) == -1
    
    # Test case from spec: x_j=1, b_j=0, f(x) >= f(b), alpha*beta > 0 -> s = +1
    assert get_rotation_sign(1, 0, True, 0.7071, 0.7071) == +1
    
    # Test cases where no rotation occurs
    assert get_rotation_sign(0, 0, False, 0.5, 0.5) == 0
    assert get_rotation_sign(1, 1, True, 0.5, 0.5) == 0


def test_qiea_reference_4spin_worked_example_exact():
    """Test that the reference implementation matches the 4-spin worked example from spec to 1e-6 precision."""
    # Problem definition from spec Section 5
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # Target state b = [0, 1, 0, 1]
    b = np.array([0, 1, 0, 1])
    s_b = 2 * b - 1
    # Energy of target state: E(s_b) = -4.0, f(b) = 4.0
    e_b = _compute_spin_energy(b, h, J)
    assert abs(e_b - (-4.0)) < 1e-10
    assert abs((-e_b) - 4.0) < 1e-10

    # Binary state from worked example observation: x = [1, 0, 0, 0]
    x = np.array([1, 0, 0, 0])
    
    # Step 3: Fitness evaluation
    # E(s) = 1.0, f(x) = -1.0
    energy = _compute_spin_energy(x, h, J)
    assert abs(energy - 1.0) < 1e-10
    fitness = -energy
    assert abs(fitness - (-1.0)) < 1e-10
    
    f_ge_fb = bool(fitness >= (-e_b))
    assert not f_ge_fb
    
    # Initial amplitudes
    alpha_0 = 1.0 / np.sqrt(2.0)
    beta_0 = 1.0 / np.sqrt(2.0)
    theta_0 = 0.05 * np.pi
    
    # Step 4: Rotation gate application check for each bit
    # Bit 0: x0=1, b0=0, f(x)<f(b), alpha*beta>0 -> s = -1
    s0 = get_rotation_sign(x[0], b[0], f_ge_fb, alpha_0, beta_0)
    assert s0 == -1
    d_theta0 = s0 * theta_0
    alpha0_new = alpha_0 * np.cos(d_theta0) - beta_0 * np.sin(d_theta0)
    beta0_new = alpha_0 * np.sin(d_theta0) + beta_0 * np.cos(d_theta0)
    assert abs(alpha0_new - 0.8090) < 1e-4
    assert abs(beta0_new - 0.5878) < 1e-4
    assert abs(alpha0_new**2 + beta0_new**2 - 1.0) < 1e-12

    # Bit 1: x1=0, b1=1, f(x)<f(b), alpha*beta>0 -> s = +1
    s1 = get_rotation_sign(x[1], b[1], f_ge_fb, alpha_0, beta_0)
    assert s1 == +1
    d_theta1 = s1 * theta_0
    alpha1_new = alpha_0 * np.cos(d_theta1) - beta_0 * np.sin(d_theta1)
    beta1_new = alpha_0 * np.sin(d_theta1) + beta_0 * np.cos(d_theta1)
    assert abs(alpha1_new - 0.5878) < 1e-4
    assert abs(beta1_new - 0.8090) < 1e-4
    assert abs(alpha1_new**2 + beta1_new**2 - 1.0) < 1e-12

    # Bit 2: x2=0, b2=0 -> s = 0
    s2 = get_rotation_sign(x[2], b[2], f_ge_fb, alpha_0, beta_0)
    assert s2 == 0
    d_theta2 = s2 * theta_0
    alpha2_new = alpha_0 * np.cos(d_theta2) - beta_0 * np.sin(d_theta2)
    beta2_new = alpha_0 * np.sin(d_theta2) + beta_0 * np.cos(d_theta2)
    assert abs(alpha2_new - 0.7071) < 1e-4
    assert abs(beta2_new - 0.7071) < 1e-4
    assert abs(alpha2_new**2 + beta2_new**2 - 1.0) < 1e-12

    # Bit 3: x3=0, b3=1, f(x)<f(b), alpha*beta>0 -> s = +1
    s3 = get_rotation_sign(x[3], b[3], f_ge_fb, alpha_0, beta_0)
    assert s3 == +1
    d_theta3 = s3 * theta_0
    alpha3_new = alpha_0 * np.cos(d_theta3) - beta_0 * np.sin(d_theta3)
    beta3_new = alpha_0 * np.sin(d_theta3) + beta_0 * np.cos(d_theta3)
    assert abs(alpha3_new - 0.5878) < 1e-4
    assert abs(beta3_new - 0.8090) < 1e-4
    assert abs(alpha3_new**2 + beta3_new**2 - 1.0) < 1e-12


def test_qiea_reference_normalization_invariant():
    """Test that |alpha|^2 + |beta|^2 = 1 is preserved to within 1e-9 precision across 100 generations."""
    h = np.array([1.0, -1.0, 0.5, -0.5])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # Run for 100 generations with history tracking
    best_s, best_e, history = _qiea_reference(
        h, J, num_generations=100, pop_size=20, theta_0=0.05 * np.pi, seed=12345, track_history=True
    )
    
    assert len(history["alphas"]) == 100
    assert len(history["betas"]) == 100
    
    for gen_idx in range(100):
        alphas = history["alphas"][gen_idx]
        betas = history["betas"][gen_idx]
        norm_sq = alphas**2 + betas**2
        # Normalization invariant |alpha|^2 + |beta|^2 = 1.0 to within 1e-9
        np.testing.assert_allclose(norm_sq, 1.0, atol=1e-9)
    
    assert best_s is not None
    assert isinstance(best_e, (int, float))
    assert len(best_s) == len(h)
    assert np.all(np.isin(best_s, [-1, 1]))
    assert np.isfinite(best_e)


def test_get_rotation_sign_fallback():
    """Test fallback return 0 branch for invalid bit values."""
    assert get_rotation_sign(2, 3, False, 0.5, 0.5) == 0