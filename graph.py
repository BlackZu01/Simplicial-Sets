from euclidean_distance import euclideanMetric
import numpy as np
from multiprocessing import Pool, cpu_count
import networkx as nx
import matplotlib.pyplot as plt

class SimplicialSet:
    def __init__(self, points: np.ndarray) -> None:
        self.points = points 
        self.graph = {}

    def createEdges(self, epsilon: float) -> None:
        total_processes = (cpu_count()*2) - 1
        with Pool(processes=total_processes) as pool:
            results = [pool.apply_async(self.computeDistances, args=(i, self.points, epsilon)) for i in range(self.points.shape[0])]
            for r in results:
                edges = r.get()
                self.graph.update(edges)

    def computeDistances(self, i: int, points: np.ndarray, epsilon: float) -> dict:
        edges = {}
        for k in range(points.shape[0]):
            if i != k:
                if euclideanMetric(points[i], points[k]) < epsilon:
                    edges[str(points[i])] = points[k]
        return edges
    
    def plotGraph(self) -> None:
        G = nx.Graph()
        for node in self.graph:
            G.add_node(node)
            for neighbor in self.graph[node]:
                G.add_edge(node, neighbor)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, node_color='lightblue', alpha=0.8, edgecolors='black')
        plt.show()
    

data = np.loadtxt('data/klein_bottle_pointcloud_new_900.txt')
print(type(data))
st = SimplicialSet(data)
st.createEdges(epsilon=1)
st.plotGraph()

# print(st.graph)