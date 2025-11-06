"""
SequenceGraph Usage Example

Demonstrates how to use the SequenceGraph with networkx.
"""

from navisim_isaac.navisim import SequenceGraph
from navisim_isaac.utils import setup_logging, get_logger


def main():
    """
    Main function demonstrating SequenceGraph usage.
    """
    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)

    logger.info("SequenceGraph Example")

    # Path to pickled NetworkX graph
    # The graph should be a NetworkX graph where:
    # - Nodes are sequence IDs (strings)
    # - Each node has data={'sectors': [list of sector IDs]}
    graph_path = "path/to/sequence_graph.pkl"

    try:
        # Load sequence graph
        logger.info(f"Loading sequence graph from {graph_path}")
        seq_graph = SequenceGraph(graph_path)

        # Get all sequence IDs
        sequence_ids = seq_graph.get_sequence_ids()
        logger.info(f"Found {len(sequence_ids)} sequences: {sequence_ids}")

        # Get first sequence
        if sequence_ids:
            first_seq_id = sequence_ids[0]
            logger.info(f"Working with sequence: {first_seq_id}")

            # Set as current node
            seq_graph.set_current_node(first_seq_id)

            # Get all sectors in this sequence
            sectors = seq_graph.get_sequence(first_seq_id)
            logger.info(f"Sequence {first_seq_id} has {len(sectors)} sectors")

            # Access data from first sector
            if sectors:
                first_sector = sectors[0]
                logger.info(f"First sector: {first_sector}")

                # Lazy-load height field
                logger.info("Accessing height field...")
                try:
                    height_field = first_sector.height_field
                    logger.info(f"Height field: {height_field}")
                except Exception as e:
                    logger.warning(f"Could not load height field: {e}")

                # Lazy-load USDZ
                logger.info("Accessing USDZ...")
                try:
                    usdz = first_sector.usdz
                    logger.info(f"USDZ: {usdz}")
                except Exception as e:
                    logger.warning(f"Could not load USDZ: {e}")

                # Lazy-load boundary polygon
                logger.info("Accessing boundary polygon...")
                try:
                    boundary = first_sector.boundary
                    logger.info(f"Boundary: {boundary}")
                except Exception as e:
                    logger.warning(f"Could not load boundary: {e}")

            # Navigate to next node
            logger.info("Requesting next node...")
            next_node = seq_graph.request_next_node()
            if next_node:
                logger.info(f"Next node: {next_node}")
            else:
                logger.info("No next node available")

            # Get neighbors
            neighbors = seq_graph.get_neighbors(first_seq_id)
            logger.info(f"Neighbors of {first_seq_id}: {neighbors}")

        # Access NetworkX graph methods directly
        logger.info(f"Total nodes in graph: {seq_graph.number_of_nodes()}")
        logger.info(f"Total edges in graph: {seq_graph.number_of_edges()}")

    except FileNotFoundError as e:
        logger.error(f"Graph file not found: {e}")
        logger.info("To use this example, create a NetworkX graph and save it as pickle:")
        logger.info("  import networkx as nx")
        logger.info("  import pickle")
        logger.info("  G = nx.Graph()")
        logger.info("  G.add_node('seq_001', sectors=['sector_0', 'sector_1'])")
        logger.info("  G.add_node('seq_002', sectors=['sector_0', 'sector_1', 'sector_2'])")
        logger.info("  G.add_edge('seq_001', 'seq_002')")
        logger.info("  with open('sequence_graph.pkl', 'wb') as f:")
        logger.info("      pickle.dump(G, f)")

    logger.info("Example completed")


if __name__ == "__main__":
    main()
