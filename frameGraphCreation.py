import networkx as nx
import matplotlib.pyplot as plt

def objectsToGraph(filePath):
    graphDict = {}
    with open(filePath, 'r') as f:
        prevFrame = "1"
        G = nx.DiGraph()
        graphCounter = 0
        nodeCounter = 0
        for line in f:
            split = line.split(',')
            if prevFrame != split[0]:  # create new graph
                graphDict[graphCounter] = G
                graphCounter += 1
                G = nx.DiGraph()
                nodeCounter = 0

            # add node and edges to current graph
            G.add_nodes_from([
                (nodeCounter,
                 {"class": split[1], "confidence": split[2],
                  "topLeftX": split[2], "topLeftY": split[3],
                  "bottomRightX": split[4], "bottomRightY": split[5]}),
            ])
            for i in range(0, nodeCounter):
                G.add_edge(nodeCounter, i)  # STILL NEED TO ADD EDGE WEIGHTS
            prevFrame = split[0]
            nodeCounter += 1
    return graphDict

            

objectsToGraph("yolov4_object_detection_results.txt")

