"""LangGraph wiring for the SelfRAG pipeline."""
from langgraph.graph import StateGraph, END, START
from selfrag.state import SessionState
from selfrag.nodes import enhance_query, retrieve, generate, ground
import logging


def build_graph(max_loops: int = 3):
    logging.info(f"[Graph] Building graph with max_loops={max_loops}")
    g = StateGraph(SessionState)

    # Register nodes
    logging.info("[Graph] Registering nodes: enhance_query, retrieve, generate, ground")
    g.add_node(enhance_query.node_name, enhance_query.enhance)
    g.add_node(retrieve.node_name, retrieve.retrieve)
    g.add_node(generate.node_name, generate.generate)
    g.add_node(ground.node_name, ground.ground)

    # Entry and linear edges
    g.set_entry_point(enhance_query.node_name)
    g.add_edge(enhance_query.node_name, retrieve.node_name)
    g.add_edge(retrieve.node_name, generate.node_name)
    g.add_edge(generate.node_name, ground.node_name)

    # Conditional looping edge
    def _continue_or_end(state: SessionState):
        if state.grounded or state.step >= max_loops:
            return END
        return enhance_query.node_name

    g.add_conditional_edges(
        ground.node_name,
        _continue_or_end,
        {enhance_query.node_name: enhance_query.node_name, END: END},
    )

    logging.info("[Graph] Graph compiled.")
    return g.compile() 