# LangGraph-inspired orchestrator (future-proof structure)
class Orchestrator:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, from_node, to_node):
        self.edges.setdefault(from_node, []).append(to_node)

    def add_conditional_edges(self, from_node, condition_fn):
        self.edges[from_node] = [
            (condition_fn, target) for target in self.nodes if target != from_node
        ]

    def execute_graph(self, start_node, agent, state):
        current_node = start_node
        while current_node:
            result = self.nodes[current_node](agent, state)
            state["last_output"] = result
            next_node = self._resolve_next_node(current_node, state)
            if not next_node:
                break
            current_node = next_node
        return result

    def _resolve_next_node(self, current_node, state):
        if current_node not in self.edges:
            return None
        for condition_fn, target in self.edges[current_node]:
            if condition_fn(state):
                return target
        return None
