import numpy as np
import time
from typing import Tuple, List, Optional, Callable
import scipy.stats as stats


def measure_tts(
    solver_func: Callable,
    h: np.ndarray,
    J: np.ndarray,
    target_energy: float,
    tolerance: float = 1e-3,
    num_seeds: int = 10,
    max_time_per_run: float = 10.0,
    **solver_kwargs
) -> Tuple[List[float], int]:
    """
    Measure Time-to-Target (TTS) for a solver over multiple random seeds.
    
    Args:
        solver_func: Function that solves the Ising/QUBO problem, returns (best_spin, best_energy)
        h: Linear bias vector
        J: Coupling matrix
        target_energy: Target energy to reach (we stop when energy <= target_energy + tolerance)
        tolerance: Energy tolerance for reaching target
        num_seeds: Number of random seeds to test
        max_time_per_run: Maximum time allowed per run (seconds)
        **solver_kwargs: Additional arguments to pass to solver_func
        
    Returns:
        Tuple of (tts_list, num_successful) where:
        - tts_list: List of TTS values for successful runs (in seconds)
        - num_successful: Number of runs that reached the target
    """
    tts_values = []
    
    for seed in range(num_seeds):
        start_time = time.time()
        
        try:
            # Call the solver with current seed
            solver_kwargs['seed'] = seed
            best_spin, best_energy = solver_func(h, J, **solver_kwargs)
            
            elapsed_time = time.time() - start_time
            
            # Check if target was reached
            if best_energy <= target_energy + tolerance:
                tts_values.append(elapsed_time)
                
        except Exception as e:
            # If solver fails, we don't count this as successful
            continue
            
        # Check if we exceeded max time per run
        if time.time() - start_time > max_time_per_run:
            continue
    
    return tts_values, len(tts_values)


def compute_tts_statistics(tts_list: List[float]) -> dict:
    """
    Compute statistics from TTS measurements.
    
    Args:
        tts_list: List of TTS values (in seconds)
        
    Returns:
        Dictionary with TTS statistics
    """
    if not tts_list:
        return {
            'median_tts': float('inf'),
            'mean_tts': float('inf'),
            'std_tts': 0.0,
            'percentile_25': float('inf'),
            'percentile_50': float('inf'),
            'percentile_75': float('inf'),
            'percentile_90': float('inf'),
            'percentile_95': float('inf'),
            'num_samples': 0
        }
    
    tts_array = np.array(tts_list)
    
    return {
        'median_tts': float(np.median(tts_array)),
        'mean_tts': float(np.mean(tts_array)),
        'std_tts': float(np.std(tts_array)),
        'percentile_25': float(np.percentile(tts_array, 25)),
        'percentile_50': float(np.percentile(tts_array, 50)),
        'percentile_75': float(np.percentile(tts_array, 75)),
        'percentile_90': float(np.percentile(tts_array, 90)),
        'percentile_95': float(np.percentile(tts_array, 95)),
        'num_samples': len(tts_list)
    }


def measure_solution_quality(
    solver_func: Callable,
    h: np.ndarray,
    J: np.ndarray,
    target_energy: float,
    tolerance: float = 1e-6,
    num_seeds: int = 20,
    **solver_kwargs
) -> Tuple[float, float, List[float]]:
    """
    Measure solution quality of a solver over multiple random seeds.
    
    Args:
        solver_func: Function that solves the Ising/QUBO problem, returns (best_spin, best_energy)
        h: Linear bias vector
        J: Coupling matrix
        target_energy: Target (optimal) energy to compare against
        tolerance: Tolerance for considering a solution "correct"
        num_seeds: Number of random seeds to test
        **solver_kwargs: Additional arguments to pass to solver_func
        
    Returns:
        Tuple of (success_rate, mean_absolute_gap, gaps_list) where:
        - success_rate: Fraction of runs that found solution within tolerance
        - mean_absolute_gap: Mean |E_found - E_target| across all runs
        - gaps_list: List of absolute gaps for each run
    """
    gaps = []
    successes = 0
    
    for seed in range(num_seeds):
        try:
            solver_kwargs['seed'] = seed
            _, best_energy = solver_func(h, J, **solver_kwargs)
            
            gap = abs(best_energy - target_energy)
            gaps.append(gap)
            
            if gap <= tolerance:
                successes += 1
                
        except Exception:
            # If solver fails, assign a large gap
            gaps.append(float('inf'))
    
    success_rate = successes / num_seeds if num_seeds > 0 else 0.0
    mean_absolute_gap = np.mean([g for g in gaps if not np.isinf(g)]) if gaps else float('inf')
    
    return success_rate, mean_absolute_gap, gaps


def kolmogorov_smirnov_test(
    sample1: List[float],
    sample2: List[float],
    alpha: float = 0.01
) -> Tuple[float, float, bool]:
    """
    Perform two-sample Kolmogorov-Smirnov test.
    
    Args:
        sample1: First sample of values
        sample2: Second sample of values
        alpha: Significance level (default 0.01 for p > 0.01 requirement)
        
    Returns:
        Tuple of (ks_statistic, p_value, passes_test) where:
        - ks_statistic: KS test statistic
        - p_value: P-value of the test
        - passes_test: True if p_value > alpha (distributions not significantly different)
    """
    if len(sample1) < 2 or len(sample2) < 2:
        # Not enough data for meaningful test
        return 0.0, 1.0, True
    
    # Filter out infinite values
    sample1_finite = [x for x in sample1 if not np.isinf(x)]
    sample2_finite = [x for x in sample2 if not np.isinf(x)]
    
    if len(sample1_finite) < 2 or len(sample2_finite) < 2:
        return 0.0, 1.0, True
    
    ks_stat, p_value = stats.ks_2samp(sample1_finite, sample2_finite)
    passes_test = p_value > alpha
    
    return ks_stat, p_value, passes_test