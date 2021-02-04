import networkx as nx
import matplotlib.pyplot as plt
import math

def objectsToGraph(filePath):
    graphDict = {}
    with open(filePath, 'r') as f:
        prevFrame = "1"
        G = nx.DiGraph()
        graphCounter = 0
        nodeCounter = 0
        distList = []
        distCounter = 0
        for line in f:
            split = line.split(',')
            if prevFrame != split[0]:  # create edges and new graph
                graphDict[graphCounter] = G

                for i in range(0, len(G.nodes())):  # edge creation
                    ixCenter = (float(G.nodes()[i]["topLeftX"]) + float(G.nodes()[i]["bottomRightX"]))/2
                    iyCenter = (float(G.nodes()[i]["topLeftY"]) + float(G.nodes()[i]["bottomRightY"]))/2
                    iMid = (ixCenter, iyCenter)
                    for j in range(i, len(G.nodes())):
                        jxCenter = (float(G.nodes()[j]["topLeftX"]) + float(G.nodes()[j]["bottomRightX"]))/2
                        jyCenter = (float(G.nodes()[j]["topLeftY"]) + float(G.nodes()[j]["bottomRightY"]))/2
                        jMid = (jxCenter, jyCenter)
                        if i != j:
                            x = iMid[0]-jMid[0]
                            y = iMid[1]-jMid[1]
                            dist = math.sqrt(math.pow(x,2)+math.pow(y,2))
                            G.add_edge(i, j, weight=dist)
                            G.add_edge(j, i, weight=dist)

                graphCounter += 1  # create new graph
                G = nx.DiGraph()
                nodeCounter = 0
            distList.append([split[0],split[2],split[3],split[4],split[5]])
            distCounter += 1
            # add node to current graph
            G.add_nodes_from([
                (nodeCounter,
                 {"class": split[1], "confidence": split[2],
                  "topLeftX": split[2], "topLeftY": split[3],
                  "bottomRightX": split[4], "bottomRightY": split[5]}),
            ])
            prevFrame = split[0]
            nodeCounter += 1
    return graphDict

            

dict = objectsToGraph("yolov4_object_detection_results.txt")
print(dict[1].edges.data())
nx.draw(dict[1])
plt.show()

