"""
ILLUSTRATIVE topology/transpilation demonstration only.
Do not report these toy results as official Processor A or Processor B benchmark results.
Replace the topology config with exact organizer-supplied processor definitions before official evaluation.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

OUT = Path("results")
(TABLES := OUT / "tables").mkdir(parents=True, exist_ok=True)
(FIGURES := OUT / "figures").mkdir(parents=True, exist_ok=True)

# Toy topologies selected only to demonstrate why connectivity impacts compilation.
# These are NOT the hackathon processors.
TOY_PROCESSORS = {
    "TOY_A_NOT_OFFICIAL": {
        "num_qubits": 5,
        "coupling_map": [[0,1], [1,2], [2,3], [3,4]],
        "basis_gates": ["rz", "sx", "x", "cx"],
    },
    "TOY_B_NOT_OFFICIAL": {
        "num_qubits": 5,
        "coupling_map": [[0,1], [0,2], [1,2], [1,3], [2,4], [3,4]],
        "basis_gates": ["rz", "sx", "x", "cx"],
    },
}

def circuit_metrics(qc):
    ops = qc.count_ops()
    twoq = {"cx", "cz", "ecr", "swap", "rxx", "ryy", "rzz"}
    return {
        "depth": int(qc.depth() or 0),
        "num_qubits_logical_register": int(qc.num_qubits),
        "total_ops": int(sum(ops.values())),
        "two_qubit_gate_count_estimate": int(sum(n for name,n in ops.items() if name.lower() in twoq)),
        "explicit_swap_count": int(ops.get("swap", 0)),
        "operation_counts": json.dumps({str(k): int(v) for k,v in ops.items()}),
    }

def draw_graph(name, cfg, path):
    graph = nx.Graph()
    graph.add_nodes_from(range(cfg["num_qubits"]))
    graph.add_edges_from(tuple(edge) for edge in cfg["coupling_map"])
    plt.figure(figsize=(5, 4))
    nx.draw_networkx(graph, with_labels=True, node_size=800)
    plt.title(name + " (TOY topology — NOT OFFICIAL)")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()

def main():
    # Demonstration workload; topology routing is the subject, not a C1 chemistry solver.
    qc = QuantumCircuit(5, 2)
    qc.h(0)
    qc.cx(0, 4)        # long-range logical interaction encourages routing on sparse graphs
    qc.ry(0.31, 2)
    qc.cx(2, 3)
    qc.measure([0, 4], [0, 1])

    rows = []
    compiled_circuits = {}
    for name, cfg in TOY_PROCESSORS.items():
        coupling = CouplingMap(couplinglist=cfg["coupling_map"])
        compiled = transpile(
            qc,
            basis_gates=cfg["basis_gates"],
            coupling_map=coupling,
            optimization_level=1,
            seed_transpiler=7,
        )
        compiled_circuits[name] = compiled
        metrics = circuit_metrics(compiled)
        metrics.update({
            "processor_label": name,
            "status": "ILLUSTRATIVE_TOY_NOT_OFFICIAL_A_OR_B",
            "logical_circuit_depth": int(qc.depth() or 0),
            "shots": 2048,
            "note": "Replace toy topology and rerun with official Challenge Kit processor definitions."
        })
        rows.append(metrics)
        with open(FIGURES / f"{name.lower()}_transpiled_circuit.txt", "w", encoding="utf-8") as f:
            f.write(str(compiled.draw(output="text")))

        draw_graph(name, cfg, FIGURES / f"{name.lower()}_graph.png")

        # Ideal simulation is for sanity-check only. It does not model the official processor noise.
        sim = AerSimulator()
        sim_result = sim.run(compiled, shots=2048, seed_simulator=7).result()
        counts = sim_result.get_counts()
        with open(OUT / f"{name.lower()}_demo_counts.json", "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k,v in counts.items()}, f, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "TOY_architecture_metrics_NOT_OFFICIAL.csv", index=False)

    plot_df = df.set_index("processor_label")[["depth", "two_qubit_gate_count_estimate", "explicit_swap_count"]]
    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Toy topology comparison — NOT official Processor A/B")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "TOY_architecture_comparison_NOT_OFFICIAL.png", dpi=180)
    plt.close()
    print(df[["processor_label", "depth", "two_qubit_gate_count_estimate", "explicit_swap_count", "status"]].to_string(index=False))
    print("\nNOTE: These are demo topologies, not the official hackathon processors.")

if __name__ == "__main__":
    main()
