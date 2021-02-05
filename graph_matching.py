class Graph(dict):
    def __init__(self, V: set, E: dict):
        self.V = V
        self.E = E

def MakeEdgeAdjacencyMap(V):
    adjacency_map = dict()
    for v1 in V:
        adjacency_map[v1] = dict()
        for v2 in V:
            adjacency_map[v1][v2] = None
    return adjacency_map

def FormatAdjacencyMap(E):
    string = str()
    for key1, value1 in sorted(E.items(), key = lambda pair : pair[0]):
        string += str(key1) + ':\n'
        for key2, value2 in sorted(value1.items(), key = lambda pair : pair[0]):
            string += '\t' + str(key2) + ': ' + str(value2) + '\n'
    return string    

def AddVectors(v1: tuple, v2: tuple):
    return tuple(map(sum, zip(v1, v2)))

def SubVectors(v1: tuple, v2: tuple):
    return tuple(map(lambda x : x[0] - x[1], zip(v1, v2)))

def AverageCostWorker(G, src, dst, visited_edges = set(), curr_cost = (0, 0)):
    if src == dst:
        return (curr_cost, 1)
    else:
        path_costs_sum = (0, 0)
        no_paths = 0
        for v in G.V:
            edge = (src, v)
            weight = G.E[src][v]
            if edge not in visited_edges and weight != None:
                visited_edges.add(edge)
                ret_cost, ret_count = AverageCostWorker(G, v, dst, visited_edges, AddVectors(curr_cost, weight))
                path_costs_sum = AddVectors(path_costs_sum, ret_cost)
                no_paths = no_paths + ret_count
        return (path_costs_sum, no_paths)

def AverageCost(G, src, dst):
    path_costs_sum, no_paths = AverageCostWorker(G, src, dst, set(), (0, 0))
    return tuple(map(lambda x : x / no_paths, path_costs_sum)) if no_paths else None

def InterpolateBackwardsEdgeWeights(G):
    for v1 in G.E:
        for v2 in G.E[v1]:
            if(G.E[v1][v2] != None and G.E[v2][v1] == None):
                G.E[v2][v1] = tuple(map(lambda x : -x, G.E[v1][v2]))

def InterpolateEdgeWeights(G, anchor):
    InterpolateBackwardsEdgeWeights(G)
    position_table = { anchor : (0, 0) }
    disconnected_nodes = set()
    for v in G.V:
        if v != anchor:
            pos = AverageCost(G, anchor, v)
            if pos != None:
                position_table[v] = pos
            else:
                disconnected_nodes.add(v)
    for v1 in position_table.keys():
        for v2 in position_table.keys():
                G.E[v1][v2] = SubVectors(position_table[v2], position_table[v1])
    return disconnected_nodes

V = set(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
E = MakeEdgeAdjacencyMap(V)
G = Graph(V, E)
G.E['a']['b'] = (-0.2, -1.0)
G.E['a']['c'] = ( 2.0, -0.7)
G.E['a']['f'] = ( 2.5, -0.5)
G.E['b']['c'] = ( 0.8,  0.3)
G.E['b']['d'] = ( 1.8, -0.6)
G.E['b']['e'] = ( 0.2, -1.0)
G.E['c']['d'] = (-0.2, -1.0)
G.E['c']['f'] = ( 0.4,  0.2)
G.E['c']['g'] = ( 0.6, -3.0)
G.E['d']['e'] = (-0.8, -0.3)
G.E['d']['g'] = ( 0.6, -0.5)
print('Before:')
print(FormatAdjacencyMap(G.E))
InterpolateEdgeWeights(G, 'a')
print('After x1:')
print(FormatAdjacencyMap(G.E))
InterpolateEdgeWeights(G, 'a')
print('After x2:')
print(FormatAdjacencyMap(G.E))
InterpolateEdgeWeights(G, 'a')
print('After x3:')
print(FormatAdjacencyMap(G.E))
