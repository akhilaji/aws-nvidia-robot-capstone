from typing import Any, Dict, List, Set

class Graph:
    def __init__(self, V: Set[Any], E: Dict[Any, Dict[Any, Any]]) -> None:
        self.V = V
        self.E = E

    def __str__(self) -> str:
        return format_adjacency_map(self.V, self.E)

def adjacency_map(V: Set[Any]) -> Dict[Any, Dict[Any, Any]]:
    return { v : dict() for v in V }

def format_adjacency_map(V: Set[Any], E: Dict[Any, Dict[Any, Any]]) -> str:
    s = str()
    for u in sorted(V):
        s += str(u) + ':\n'
        for v in sorted(V):
            s += '\t' + str(v) + ': '
            s += str(E.get(u).get(v) if u in E else None)
            s += '\n'
    return s

def all_simple_paths_recursive(G: Graph, src: Any, dst: Any) -> List[Any]:
    def worker(G: Graph, src: Any, dst: Any, path: List[Any], visited: Set[Any]) -> List[Any]:
        if src == dst:
            return [path.copy()]
        else:
            paths = []
            for node in G.E[src].keys():
                if node not in visited:
                    path.append(node)
                    visited.add(node)
                    paths.extend(worker(G, node, dst, path, visited))
                    visited.remove(node)
                    path.pop()
            return paths
    return worker(G, src, dst, [src], {src})

def all_simple_paths_iterative(G: Graph, src: Any, dst: Any) -> List[Any]:
    paths = []
    path = [src]
    visited = {src}
    fringe = [(src, iter(G.E[src].keys()))]
    while fringe:
        node, itr = fringe[-1]
        if node == dst:
            paths.append(path.copy())
            visited.remove(node)
            fringe.pop()
            path.pop()
        else:
            try:
                node = next(itr)
                if node not in visited:
                    visited.add(node)
                    fringe.append((node, iter(G.E[node].keys())))
                    path.append(node)
            except:
                visited.remove(node)
                fringe.pop()
                path.pop()
    return paths

def all_simple_path_costs_recursive(G: Graph, src: Any, dst: Any, initial_cost: Any) -> List[Any]:
    def worker(G: Graph, src: Any, dst: Any, cost: Any, visited: Set[Any]) -> List[Any]:
        if src == dst:
            return [cost]
        else:
            costs = []
            for node in G.E[src].keys():
                if node not in visited:
                    visited.add(node)
                    costs.extend(worker(G, node, dst, cost + G.E[src][node], visited))
                    visited.remove(node)
            return costs
    return worker(G, src, dst, initial_cost, {src})

def all_simple_path_costs_iterative(G: Graph, src: Any, dst: Any, initial_cost: Any) -> List[Any]:
    costs = []
    cost = initial_cost
    fringe = [(src, iter(G.E[src].keys()))]
    visited = { src : src }
    while fringe:
        curr, itr = fringe[-1]
        if curr == dst:
            costs.append(cost)
            cost = cost - G.E[visited[curr]][curr]
            visited.pop(curr)
            fringe.pop()
        else:
            try:
                node = next(itr)
                if node not in visited:
                    cost = cost + G.E[curr][node]
                    visited[node] = curr
                    fringe.append((node, iter(G.E[node].keys())))
            except:
                if visited[curr] != curr:
                    cost = cost - G.E[visited[curr]][curr]
                visited.pop(curr)
                fringe.pop()
    return costs

all_simple_paths = all_simple_paths_iterative
all_simple_path_costs = all_simple_path_costs_iterative
