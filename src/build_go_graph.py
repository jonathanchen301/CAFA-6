import networkx as nx
import pickle
import argparse
import os

def parse_obo(path: str) -> list[dict]:
    """
    Parse the OBO file and return a list of terms

    Args:
    - path: Path to the OBO file
    
    Returns:
    - terms: List of terms

    Example:
    ```
    [
    {
        "id": "GO:0000001",
        "is_a": ["GO:0000002", "GO:0000003"],
        "part_of": ["GO:0000004", "GO:0000005"],
        "is_obsolete": False,
    },
    ...,
    ]
    ```
    """
    with open(path, "r") as file:
        terms = []
        current_term = None
        for line in file:
            line = line.strip()

            if line == "[Term]":
                if current_term and not current_term["is_obsolete"]:
                    terms.append(current_term)
                current_term = {
                    "id": None,
                    "is_a": [],
                    "part_of": [],
                    "is_obsolete": False,
                }

            # Reset current_term when encountering other section types (like [Typedef])
            elif line.startswith("[") and line.endswith("]"):
                if current_term and not current_term["is_obsolete"]:
                    terms.append(current_term)
                current_term = None
            # Skips lines before the first [Term]
            elif current_term is None:
                continue

            elif line.startswith("id:"):
                current_term["id"] = line.split(":", 1)[1].strip()
            elif line.startswith("is_a:"):
                go_id = line.split(":", 1)[1].split("!")[0].strip()
                current_term["is_a"].append(go_id)
            elif line.startswith("relationship: part_of"):
                go_id = line.split("part_of")[1].split("!")[0].strip()
                current_term["part_of"].append(go_id)
            elif line.startswith("is_obsolete:"):
                current_term["is_obsolete"] = line.split(":", 1)[1].strip().lower() == "true"
            
        # Include the last term
        if current_term and not current_term["is_obsolete"]:
            terms.append(current_term)

        return terms
    
def build_go_graph(terms: list[dict]) -> nx.DiGraph:
    """
    Build the GO graph from the list of terms extracted

    Args:
    - terms: List of terms extracted from the OBO file

    Returns:
    - G: NetworkX directed graph
    """
    # IDs in which it is actually a [term] inside the OBO file
    valid_ids = set(term["id"] for term in terms)

    G = nx.DiGraph()
    for term in terms:
        G.add_node(term["id"])
        for parent in term["is_a"]:
            if parent in valid_ids:
                G.add_edge(term["id"], parent, relationship_type="is_a")
        for part in term["part_of"]:
            if part in valid_ids:
                G.add_edge(term["id"], part, relationship_type="part_of")
    return G

def save_go_graph(G: nx.DiGraph, path: str) -> None:
    """
    Save GO graph using pickle

    Args:
    - G: NetworkX directed graph
    - path: Path to save the GO graph

    Returns:
    - None
    """

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, "wb") as file:
        pickle.dump(G, file)
    print("GO graph saved to:", path)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

def main():
    parser = argparse.ArgumentParser(
        description="Parse a GO OBO file and build a NetworkX GO graph."
    )

    parser.add_argument("--obo_path", type=str, required=True, help="Path to the GO OBO file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the GO graph")

    args = parser.parse_args()

    terms = parse_obo(args.obo_path)
    G = build_go_graph(terms)
    save_go_graph(G, args.output_path)

if __name__ == "__main__":
    main()