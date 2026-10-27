"""
Golden instances generator for QIEA algorithm verification.

This module generates reference optimal instances for testing the
Quantum-Inspired Evolutionary Algorithm (QIEA).

Per spec Section 9, provides:
- 4 planted instances with known optima (brute-force verified)
- 2 MQLib-style instances with proven optimal solutions
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def compute_isin_energy(h: np.ndarray, J: np.ndarray, spins: np.ndarray) -> float:
    """
    Independent first-principles Ising energy calculation.
    
    Computes E(s) = sum_i h_i * s_i + sum_{i<j} J_ij * s_i * s_j
    
    Args:
        h: Linear bias vector (N,)
        J: Coupling matrix (N x N), symmetric, zero diagonal
        spins: Ising spin vector (N,) with values in {-1, +1}
    
    Returns:
        Energy value (float)
    """
    h_arr = np.asarray(h, dtype=np.float64)
    J_arr = np.asarray(J, dtype=np.float64)
    s_arr = np.asarray(spins, dtype=np.float64)
    N = len(h_arr)
    
    linear = float(np.dot(h_arr, s_arr))
    quad = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            quad += float(J_arr[i, j] * s_arr[i] * s_arr[j])
    return linear + quad


def brute_force_search(h: np.ndarray, J: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Independent brute-force search over all 2^N states.
    
    Args:
        h: Linear bias vector (N,)
        J: Coupling matrix (N x N)
    
    Returns:
        Tuple of (optimal_spins, optimal_energy)
    
    Raises:
        ValueError: If N > 16 (computation infeasible)
    """
    h_arr = np.asarray(h, dtype=np.float64)
    J_arr = np.asarray(J, dtype=np.float64)
    N = len(h_arr)
    
    if N > 16:
        raise ValueError(f"Brute force limited to N <= 16, got N={N}")
    
    min_energy = float("inf")
    best_spins = None
    
    for idx in range(1 << N):
        # Convert binary index to Ising spins {-1, +1}
        spins = np.array([1.0 if ((idx >> k) & 1) else -1.0 for k in range(N)], dtype=np.float64)
        energy = compute_isin_energy(h_arr, J_arr, spins)
        
        if energy < min_energy:
            min_energy = energy
            best_spins = spins.copy()
    
    return best_spins, min_energy


def generate_planted_instance(h: np.ndarray, J: np.ndarray) -> Dict[str, Any]:
    """
    Generate a planted instance with exact optimum.
    
    Uses brute-force to find the exact optimum.
    """
    best_spins, best_energy = brute_force_search(h, J)
    N = len(h)
    
    return {
        "h": h.tolist(),
        "J": J.tolist(),
        "N": N,
        "certified_optimum": {
            "energy": float(best_energy),
            "spins": best_spins.tolist(),
            "method": "brute_force_exact"
        }
    }


def load_golden_instances(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load golden instances from JSON file.
    
    Args:
        path: Path to golden_instances.json. If None, uses default location.
    
    Returns:
        Dictionary with 'instances' key containing instance data.
    
    Raises:
        FileNotFoundError: If golden instances file doesn't exist.
    """
    if path is None:
        # Default: look in golden/ directory relative to this module
        candidates = [
            Path(__file__).resolve().parent / "golden_instances.json",
            Path(__file__).resolve().parent / "golden" / "golden_instances.json",
            Path(__file__).resolve().parent.parent / "golden" / "golden_instances.json",
            Path.cwd() / "golden" / "golden_instances.json",
            Path.cwd() / "tests" / "fixtures" / "golden" / "golden_instances.json",
            Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden" / "golden_instances.json",
        ]
        path = next((c for c in candidates if c.exists()), Path(__file__).resolve().parent / "golden_instances.json")
    else:
        path = Path(path)
        if not path.exists():
            candidates = [
                Path(__file__).resolve().parent / path.name,
                Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden" / path.name,
                Path.cwd() / "tests" / "fixtures" / "golden" / path.name,
            ]
            path = next((c for c in candidates if c.exists()), path)
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden instances file not found at {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    return data


def compute_optimality_gap(energy_found: float, energy_opt: float) -> float:
    """
    Compute the absolute optimality gap.
    
    Args:
        energy_found: Energy found by solver
        energy_opt: Reference optimal energy
    
    Returns:
        Absolute gap |E_found - E_opt|
    """
    return abs(energy_found - energy_opt)


def validate_against_certified(
    instance: Dict[str, Any],
    solver_result: Tuple[np.ndarray, float],
    tolerance_planted: float = 1e-4,
    tolerance_mqlib: float = 0.01
) -> Tuple[bool, float, str]:
    """
    Validate solver result against reference optimum.
    
    Args:
        instance: Instance dictionary with 'certified_optimum' key
        solver_result: Tuple of (best_spins, best_energy) from solver
        tolerance_planted: Tolerance for planted instances
        tolerance_mqlib: Tolerance for MQLib instances
    
    Returns:
        Tuple of (is_valid, gap, message)
    """
    h = np.array(instance["h"], dtype=np.float64)
    J = np.array(instance["J"], dtype=np.float64)
    
    best_spins, best_energy = solver_result
    certified = instance["certified_optimum"]
    certified_energy = certified["energy"]
    
    # Verify the returned spins have a valid configuration
    if not np.all(np.isin(best_spins, [-1.0, 1.0])):
        return False, float("inf"), "Invalid spin values in result"
    
    # Compute gap
    gap = compute_optimality_gap(best_energy, certified_energy)
    
    # Check against reference optimum independently
    check_energy = compute_isin_energy(h, J, best_spins.astype(np.float64))
    if abs(check_energy - best_energy) > 1e-9:
        return False, float("inf"), f"Solver energy mismatch: {best_energy} vs computed {check_energy}"
    
    # Determine tolerance based on instance type
    instance_type = instance.get("type", "planted")
    if instance_type == "planted":
        tolerance = tolerance_planted
    else:
        tolerance = tolerance_mqlib
    
    is_valid = gap <= tolerance
    
    if is_valid:
        msg = f"Valid: gap={gap:.6e} <= tolerance={tolerance}"
    else:
        msg = f"Invalid: gap={gap:.6e} > tolerance={tolerance}"
    
    return is_valid, gap, msg


def get_golden_instance_names() -> Dict[str, str]:
    """
    Get mapping of instance names to their types.
    
    Returns:
        Dictionary mapping instance names to types ('planted' or 'mqlib')
    """
    return {
        "planted_4spin": "planted",
        "planted_6spin": "planted",
        "planted_8spin": "planted",
        "planted_10spin": "planted",
        "mqlib_rudy_10": "mqlib",
        "mqlib_rudy_12": "mqlib"
    }


class GoldenInstances:
    """
    Container for all golden instances with validation methods.
    """
    
    def __init__(self, path: Optional[Path] = None):
        """Load golden instances from file."""
        self.data = load_golden_instances(path)
        self.instances = self.data.get("instances", {})
    
    def __getitem__(self, name: str) -> Dict[str, Any]:
        """Get instance by name."""
        return self.instances[name]
    
    def __iter__(self):
        """Iterate over instances."""
        return iter(self.instances.values())
    
    def __len__(self) -> int:
        """Number of instances."""
        return len(self.instances)
    
    def validate(
        self,
        instance_name: str,
        solver_result: Tuple[np.ndarray, float],
        tolerance_planted: float = 1e-4,
        tolerance_mqlib: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Validate solver result for a specific instance.
        
        Args:
            instance_name: Name of the instance to validate
            solver_result: Tuple of (best_spins, best_energy) from solver
            tolerance_planted: Tolerance for planted instances
            tolerance_mqlib: Tolerance for MQLib instances
        
        Returns:
            Tuple of (is_valid, gap)
        """
        if instance_name not in self.instances:
            raise ValueError(f"Unknown instance: {instance_name}")
        
        return validate_against_certified(
            self.instances[instance_name],
            solver_result,
            tolerance_planted,
            tolerance_mqlib
        )
    
    def validate_all(
        self,
        solver_func,
        num_runs: int = 5,
        tolerance_planted: float = 1e-4,
        tolerance_mqlib: float = 0.01
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate solver on all instances.
        
        Args:
            solver_func: Function that takes (h, J) and returns (spins, energy)
            num_runs: Number of runs per instance to check
            tolerance_planted: Tolerance for planted instances
            tolerance_mqlib: Tolerance for MQLib instances
        
        Returns:
            Dictionary mapping instance names to validation results
        """
        results = {}
        
        for name, instance in self.instances.items():
            h = np.array(instance["h"], dtype=np.float64)
            J = np.array(instance["J"], dtype=np.float64)
            cert_energy = instance["certified_optimum"]["energy"]
            
            gaps = []
            valid_runs = 0
            
            for run in range(num_runs):
                seed = (abs(hash(name)) + run * 1000) % (2**32)
                best_s, best_e = solver_func(h, J, seed=seed)
                gap = compute_optimality_gap(best_e, cert_energy)
                gaps.append(gap)
                
                is_valid, _, _ = validate_against_certified(
                    instance, (best_s, best_e), tolerance_planted, tolerance_mqlib
                )
                if is_valid:
                    valid_runs += 1
            
            results[name] = {
                "type": instance.get("type", "unknown"),
                "certified_energy": cert_energy,
                "mean_gap": float(np.mean(gaps)),
                "min_gap": float(np.min(gaps)),
                "valid_runs": valid_runs,
                "total_runs": num_runs,
                "success_rate": valid_runs / num_runs
            }
        
        return results


def main():
    """Generate and save golden instances."""
    instances = {}
    
    # 1. planted_4spin (Worked example from spec Section 5)
    h_4 = np.array([1.0, -1.0, 0.5, -0.5])
    J_4 = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    s_4, e_4 = brute_force_search(h_4, J_4)
    # Verify against spec
    assert abs(e_4 - (-4.0)) < 1e-9, f"4-spin energy mismatch: {e_4}"
    
    instances["planted_4spin"] = {
        "name": "planted_4spin",
        "type": "planted",
        "N": 4,
        "h": h_4.tolist(),
        "J": J_4.tolist(),
        "certified_optimum": {
            "energy": -4.0,
            "spins": s_4.tolist(),
            "method": "brute_force_exact"
        }
    }
    
    # 2. planted_6spin (N=6 planted instance)
    rng6 = np.random.RandomState(42)
    J_6 = rng6.uniform(-1.0, 1.0, size=(6, 6))
    J_6 = np.round((J_6 + J_6.T) / 2.0, 2)
    np.fill_diagonal(J_6, 0.0)
    h_6 = np.round(rng6.uniform(-1.5, 1.5, size=6), 2)
    s_6, e_6 = brute_force_search(h_6, J_6)
    
    instances["planted_6spin"] = {
        "name": "planted_6spin",
        "type": "planted",
        "N": 6,
        "h": h_6.tolist(),
        "J": J_6.tolist(),
        "certified_optimum": {
            "energy": float(e_6),
            "spins": s_6.tolist(),
            "method": "brute_force_exact"
        }
    }
    
    # 3. planted_8spin (N=8 planted instance per spec Section 9)
    rng8 = np.random.RandomState(88)
    J_8 = rng8.uniform(-1.0, 1.0, size=(8, 8))
    J_8 = np.round((J_8 + J_8.T) / 2.0, 2)
    np.fill_diagonal(J_8, 0.0)
    h_8 = np.round(rng8.uniform(-1.5, 1.5, size=8), 2)
    s_8, e_8 = brute_force_search(h_8, J_8)
    
    instances["planted_8spin"] = {
        "name": "planted_8spin",
        "type": "planted",
        "N": 8,
        "h": h_8.tolist(),
        "J": J_8.tolist(),
        "certified_optimum": {
            "energy": float(e_8),
            "spins": s_8.tolist(),
            "method": "brute_force_exact"
        }
    }
    
    # 4. planted_10spin (N=10 planted instance)
    rng10 = np.random.RandomState(101)
    J_10 = rng10.uniform(-1.0, 1.0, size=(10, 10))
    J_10 = np.round((J_10 + J_10.T) / 2.0, 2)
    np.fill_diagonal(J_10, 0.0)
    h_10 = np.round(rng10.uniform(-1.5, 1.5, size=10), 2)
    s_10, e_10 = brute_force_search(h_10, J_10)
    
    instances["planted_10spin"] = {
        "name": "planted_10spin",
        "type": "planted",
        "N": 10,
        "h": h_10.tolist(),
        "J": J_10.tolist(),
        "certified_optimum": {
            "energy": float(e_10),
            "spins": s_10.tolist(),
            "method": "brute_force_exact"
        }
    }
    
    # 5. mqlib_rudy_10 (N=10 MQLib-style graph QUBO)
    rng_mq1 = np.random.RandomState(201)
    adj_10 = (rng_mq1.rand(10, 10) < 0.5).astype(float)
    adj_10 = np.triu(adj_10, 1)
    weights_10 = rng_mq1.choice([-1.0, 1.0], size=(10, 10))
    J_mq1 = adj_10 * weights_10
    J_mq1 = J_mq1 + J_mq1.T
    np.fill_diagonal(J_mq1, 0.0)
    h_mq1 = np.zeros(10)
    s_mq1, e_mq1 = brute_force_search(h_mq1, J_mq1)
    
    instances["mqlib_rudy_10"] = {
        "name": "mqlib_rudy_10",
        "type": "mqlib",
        "N": 10,
        "h": h_mq1.tolist(),
        "J": J_mq1.tolist(),
        "certified_optimum": {
            "energy": float(e_mq1),
            "spins": s_mq1.tolist(),
            "method": "brute_force_exact"
        }
    }
    
    # 6. mqlib_rudy_12 (N=12 MQLib-style graph QUBO)
    rng_mq2 = np.random.RandomState(202)
    adj_12 = (rng_mq2.rand(12, 12) < 0.4).astype(float)
    adj_12 = np.triu(adj_12, 1)
    weights_12 = rng_mq2.choice([-1.0, 1.0], size=(12, 12))
    J_mq2 = adj_12 * weights_12
    J_mq2 = J_mq2 + J_mq2.T
    np.fill_diagonal(J_mq2, 0.0)
    h_mq2 = np.zeros(12)
    s_mq2, e_mq2 = brute_force_search(h_mq2, J_mq2)
    
    instances["mqlib_rudy_12"] = {
        "name": "mqlib_rudy_12",
        "type": "mqlib",
        "N": 12,
        "h": h_mq2.tolist(),
        "J": J_mq2.tolist(),
        "certified_optimum": {
            "energy": float(e_mq2),
            "spins": s_mq2.tolist(),
            "method": "brute_force_exact"
        }
    }
    
    return {"instances": instances}


if __name__ == "__main__":
    data = main()
    
    golden_dir = Path(__file__).resolve().parent.parent / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    
    golden_path = golden_dir / "golden_instances.json"
    with open(golden_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote golden instances to {golden_path}")