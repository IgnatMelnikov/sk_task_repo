import random
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt


class Graph(object):
    def __init__(self):
        self.linked_nodes = {}
        self.nodes_info = {}

    def add_node(self, node_name, node_info):
        self.linked_nodes[node_name] = []
        self.nodes_info[node_name] = node_info

    def add_edge(self, first_node, second_node):
        if (first_node not in self.linked_nodes.keys() or
                second_node not in self.linked_nodes.keys()):
            raise ValueError('Some of nodes not in graph')
        if (first_node not in self.linked_nodes[second_node] and
                second_node not in self.linked_nodes[first_node]):
            self.linked_nodes[first_node].append(second_node)
            if first_node != second_node:
                self.linked_nodes[second_node].append(first_node)

    def create_random_graph(self, nodes_number, edge_prob=0.5):
        for i in range(nodes_number):
            self.add_node('node_{}'.format(i), 'ololo_{}'.format(i))
        for i in range(nodes_number):
            for j in range(i + 1, nodes_number):
                if random.uniform(0, 1) > edge_prob:
                    self.add_edge('node_{}'.format(i),
                                  'node_{}'.format(j))

    def draw(self, filename=None):
        G = nx.Graph()
        plt.figure()
        for u in self.linked_nodes.keys():
            G.add_node(u)
            for v in self.linked_nodes[u]:
                G.add_edge(u, v)
        nx.draw_networkx(G)
        if filename is not None:

            plt.savefig(filename)

        else:
            plt.show()
