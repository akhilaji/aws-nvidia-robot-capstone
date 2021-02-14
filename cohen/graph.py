"""
This module defines the Graph class and utility functions for its usage in
reconstructing scenes.



"""

from typing import Dict

class Graph:
    """
    Container class for the set of nodes V, and edge adjacency dictionary E.
    The edge adjacency dictionary E maps each node v1 in V to another
    dictionary where for each node v2 in V that if there exists an edge
    spanning from v1 to v2 then v2 is mapped to that edge's weight.
    """

    def __init__(self, V: set, E: dict):
        self.V = V
        self.E = E

    def __str__(self):
        return format_adjacency_map(self.V, self.E)

def adjacency_map(V: set) -> Dict[Dict]:
    """
    Constructs an empty edge adjacency map. Meaning it constructs a dictionary
    where each node in V is mapped to an empty dictionary.
    """
    return { v : dict() for v in V }

def format_adjacency_map(V: set, E: dict) -> str:
    s = str()
    for u in sorted(V):
        s += str(u) + ':\n'
        for v in sorted(V):
            s += '\t' + str(v) + ': '
            s += str(E.get(u).get(v) if u in E else None)
            s += '\n'
    return s

def all_simple_paths_recursive(G: Graph, src, dst) -> list:
    def worker(G: Graph, src, dst, path: list, visited: set) -> list:
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

def all_simple_paths_iterative(G: Graph, src, dst) -> list:
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

def all_simple_path_costs_recursive(G: Graph, src, dst, initial_cost) -> list:
    def worker(G: Graph, src, dst, cost, visited: set) -> list:
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

def all_simple_path_costs_iterative(G: Graph, src, dst, initial_cost) -> list:
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

def average_simple_path_cost_recursive(G: Graph, src, dst, initial_cost):
    costs = all_simple_path_costs_recursive(G, src, dst, initial_cost)
    return sum(costs) / len(costs)

def average_simple_path_cost_iterative(G: Graph, src, dst, initial_cost):
    costs = all_simple_path_costs_iterative(G, src, dst, initial_cost)
    return sum(costs) / len(costs)
