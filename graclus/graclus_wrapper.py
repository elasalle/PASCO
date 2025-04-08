import networkx as nx
import tempfile
import subprocess
from pathlib import Path
from typing import Dict

def write_graclus_input(graph: nx.Graph, filepath: Path):
    with open(filepath, 'w') as f:
        f.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for node in sorted(graph.nodes()):
            neighbors = sorted(graph.neighbors(node))
            if neighbors:
                f.write(' '.join(str(n + 1) for n in neighbors))  # 1-based index
            f.write('\n')

def read_clustering_output(filepath: Path) -> Dict[int, int]:
    with open(filepath, 'r') as f:
        return {i: int(line.strip()) for i, line in enumerate(f)}

# graclus_path = "C:\\Users\\adminuser\\Documents\\GitHub\\PASCO\\graclus\\graclus1.2\\graclus.exe"
graclus_path = "/projects/users/elasalle/PASCO/graclus/graclus1.2/graclus.exe"
def run_graclus(graph: nx.Graph, num_clusters: int, graclus_exe_path: str = graclus_path) -> Dict[int, int]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_file = tmpdir_path / "input.graph"
        output_file = tmpdir_path / f"input.graph.part.{num_clusters}"

        write_graclus_input(graph, input_file)

        result = subprocess.run([
            graclus_exe_path, str(input_file), str(num_clusters)
        ], capture_output=True, text=True, cwd=tmpdir_path)

        if result.returncode != 0:
            raise RuntimeError(f"Graclus failed: {result.stderr}")

        return read_clustering_output(output_file)
