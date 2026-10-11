"""
PS-C1 H2 ground-state energy estimation using Qiskit Nature VQE.
Starter baseline; adapt to exact C1 Challenge Kit and Processor A/B definitions before submission.
"""
from __future__ import annotations
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

def run_h2_vqe():
    try:
        import qiskit
        import qiskit_nature
        import qiskit_algorithms
        from qiskit.primitives import StatevectorEstimator
        from qiskit_algorithms import VQE, NumPyMinimumEigensolver
        from qiskit_algorithms.optimizers import SLSQP
        from qiskit_nature.units import DistanceUnit
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
        from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install requirements.txt in a fresh Python environment. "
            f"Original error: {exc}"
        )

    cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
    bond = float(cfg["bond_length_angstrom"])
    seed = int(cfg["seed"])
    np.random.seed(seed)

    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {bond}",
        basis=cfg["basis"],
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

    exact_solver = GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
    t0 = time.perf_counter()
    exact_result = exact_solver.solve(problem)
    exact_runtime = time.perf_counter() - t0
    exact_total = float(np.real(exact_result.total_energies[0]))

    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=hf,
    )
    history = []
    def callback(eval_count, parameters, mean, metadata):
        history.append(float(np.real(mean)))

    optimizer = SLSQP(maxiter=250, ftol=1e-9)
    vqe = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )
    vqe.initial_point = np.zeros(ansatz.num_parameters)

    t0 = time.perf_counter()
    # VQE's eigenvalue is electronic energy. GroundStateEigensolver applies the chemistry interpretation
    # and includes nuclear repulsion for total energy.
    vqe_solver = GroundStateEigensolver(mapper, vqe)
    vqe_result = vqe_solver.solve(problem)
    vqe_runtime = time.perf_counter() - t0
    vqe_total = float(np.real(vqe_result.total_energies[0]))
    abs_err = abs(vqe_total - exact_total)

    result = {
        "problem_code": "C1",
        "problem": "H2 molecular ground-state energy",
        "basis": cfg["basis"],
        "bond_length_angstrom": bond,
        "method": "VQE/UCCSD",
        "simulator": "StatevectorEstimator (ideal statevector)",
        "status": "LOCAL_SIMULATOR_BASELINE_NOT_PROCESSOR_A_B",
        "num_spatial_orbitals": int(problem.num_spatial_orbitals),
        "num_particles": [int(x) for x in problem.num_particles],
        "num_qubits": int(ansatz.num_qubits),
        "ansatz_parameters": int(ansatz.num_parameters),
        "vqe_total_energy_hartree": vqe_total,
        "exact_total_energy_hartree": exact_total,
        "absolute_error_hartree": abs_err,
        "absolute_error_mhartree": abs_err * 1000.0,
        "vqe_runtime_seconds": vqe_runtime,
        "exact_runtime_seconds": exact_runtime,
        "seed": seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit_version": qiskit.__version__,
        "qiskit_nature_version": qiskit_nature.__version__,
        "qiskit_algorithms_version": qiskit_algorithms.__version__,
        "warning": "Ideal simulator result; not an official Processor A/B result."
    }
    (OUT / "h2_vqe_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(TABLES / "h2_vqe_metrics.csv", index=False)

    if history:
        plt.figure(figsize=(8, 4.8))
        plt.plot(np.arange(1, len(history)+1), history, marker=".", linewidth=1, label="VQE callback energy")
        plt.axhline(exact_total, linestyle="--", label="Exact total-energy reference")
        plt.xlabel("Optimizer evaluation")
        plt.ylabel("Energy (Hartree)")
        plt.title("H₂ VQE convergence (ideal simulator baseline)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / "h2_vqe_convergence.png", dpi=180)
        plt.close()

    print(json.dumps(result, indent=2))
    print("\nSaved results/tables/h2_vqe_metrics.csv and results/h2_vqe_result.json")
    print("This is a local ideal-simulator baseline only. Add official Processor A/B experiments.")

if __name__ == "__main__":
    run_h2_vqe()
