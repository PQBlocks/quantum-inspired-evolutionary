import json
import os
import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NOTE: Per MEMORY.md (lines 43-45), test oracles MUST be independent:
# Re-derive energy from first principles; DO NOT import energy function from solver code.
def independent_ising_energy(h, J, spins):
    """Independent first-principles Ising energy oracle: sum_i h_i*s_i + sum_{i<j} J_ij*s_i*s_j"""
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


def independent_brute_force(h, J):
    """Independent brute-force search over all 2^N states."""
    h_arr = np.asarray(h, dtype=np.float64)
    J_arr = np.asarray(J, dtype=np.float64)
    N = len(h_arr)
    assert N <= 16, "Brute force limited to N <= 16"
    
    min_energy = float("inf")
    best_spins = None
    
    for idx in range(1 << N):
        spins = np.array([1.0 if ((idx >> k) & 1) else -1.0 for k in range(N)], dtype=np.float64)
        e = independent_ising_energy(h_arr, J_arr, spins)
        if e < min_energy:
            min_energy = e
            best_spins = spins.copy()
            
    return best_spins, min_energy


_fcand = Path(__file__).resolve().parent / "fixtures" / "golden" / "golden_instances.json"
_pcand = Path(__file__).resolve().parent.parent / "src" / "pqblocks_qiea" / "golden_instances.json"
_rcand = Path(__file__).resolve().parent.parent / "golden" / "golden_instances.json"
GOLDEN_PATH = _fcand if _fcand.exists() else (_pcand if _pcand.exists() else _rcand)
_ifcand = Path(__file__).resolve().parent / "fixtures" / "golden" / "instances.json"
_ipcand = Path(__file__).resolve().parent.parent / "src" / "pqblocks_qiea" / "instances.json"
_ircand = Path(__file__).resolve().parent.parent / "golden" / "instances.json"
INSTANCES_PATH = _ifcand if _ifcand.exists() else (_ipcand if _ipcand.exists() else (_ircand if _ircand.exists() else GOLDEN_PATH))


def test_golden_file_exists():
    """Verify that golden instance JSON files exist on disk."""
    assert GOLDEN_PATH.exists(), f"Golden file not found at {GOLDEN_PATH}"
    assert INSTANCES_PATH.exists(), f"Instances file not found at {INSTANCES_PATH}"


def test_golden_instance_schema_and_count():
    """Verify schema, constraints, and count of golden instances."""
    with open(GOLDEN_PATH, "r") as f:
        data = json.load(f)
        
    assert "instances" in data
    instances = data["instances"]
    
    # Spec requires 4 planted + 2 MQLib = 6 instances
    assert len(instances) == 6, f"Expected 6 golden instances, found {len(instances)}"
    
    planted_count = sum(1 for inst in instances.values() if inst["type"] == "planted")
    mqlib_count = sum(1 for inst in instances.values() if inst["type"] == "mqlib")
    assert planted_count == 4, f"Expected 4 planted instances, found {planted_count}"
    assert mqlib_count == 2, f"Expected 2 MQLib instances, found {mqlib_count}"
    
    for name, inst in instances.items():
        assert "name" in inst
        assert "N" in inst
        assert "h" in inst
        assert "J" in inst
        assert "certified_optimum" in inst
        
        N = inst["N"]
        assert N <= 20, f"Instance {name} has N={N} > 20"
        h = np.array(inst["h"], dtype=np.float64)
        J = np.array(inst["J"], dtype=np.float64)
        
        assert h.shape == (N,)
        assert J.shape == (N, N)
        assert np.allclose(J, J.T), f"J is not symmetric in {name}"
        assert np.all(np.diag(J) == 0.0), f"J has non-zero diagonal in {name}"
        
        cert = inst["certified_optimum"]
        assert "energy" in cert
        assert "spins" in cert
        assert "method" in cert
        assert len(cert["spins"]) == N
        assert set(cert["spins"]).issubset({-1, 1, -1.0, 1.0})


def test_certified_energies_match_independent_oracle():
    """Verify each reference optimal state matches independent first-principles energy oracle."""
    with open(GOLDEN_PATH, "r") as f:
        instances = json.load(f)["instances"]
        
    for name, inst in instances.items():
        h = inst["h"]
        J = inst["J"]
        cert = inst["certified_optimum"]
        spins = cert["spins"]
        claimed_energy = cert["energy"]
        
        recalc_energy = independent_ising_energy(h, J, spins)
        assert abs(recalc_energy - claimed_energy) < 1e-9, (
            f"Instance {name}: claimed energy {claimed_energy} != recalculated {recalc_energy}"
        )


def test_planted_4spin_worked_example_brute_force():
    """Independent brute force on 4-spin worked example confirms -4.0 optimum."""
    with open(GOLDEN_PATH, "r") as f:
        inst = json.load(f)["instances"]["planted_4spin"]
        
    best_spins, min_energy = independent_brute_force(inst["h"], inst["J"])
    assert abs(min_energy - (-4.0)) < 1e-9
    assert abs(min_energy - inst["certified_optimum"]["energy"]) < 1e-9


def test_all_instances_brute_force_verified():
    """Verify that independent brute force confirms all 6 instances reach claimed optimum."""
    with open(GOLDEN_PATH, "r") as f:
        instances = json.load(f)["instances"]
        
    for name, inst in instances.items():
        N = inst["N"]
        claimed_energy = inst["certified_optimum"]["energy"]
        best_spins, min_energy = independent_brute_force(inst["h"], inst["J"])
        assert abs(min_energy - claimed_energy) < 1e-6, (
            f"Instance {name} (N={N}): brute-force min energy {min_energy} != claimed {claimed_energy}"
        )


def test_qiea_solver_on_golden_4spin():
    """Verify QIEA solver reaches the reference optimum on planted_4spin."""
    from pqblocks_qiea.qiea_api import solve_ising
    
    with open(GOLDEN_PATH, "r") as f:
        inst = json.load(f)["instances"]["planted_4spin"]
        
    h = np.array(inst["h"], dtype=np.float64)
    J = np.array(inst["J"], dtype=np.float64)
    certified_opt = inst["certified_optimum"]["energy"]
    
    best_s, best_e = solve_ising(h, J, num_generations=50, population_size=10, seed=42)
    assert abs(best_e - certified_opt) < 1e-9, f"Solver energy {best_e} did not match optimum {certified_opt}"
