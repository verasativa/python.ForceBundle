import ForceBundleFunctionalNumba
from functools import partial
import networkx as nx
from xml.dom.minidom import parse
import xml.dom.minidom
from numba import jitclass, int16
import numpy as np

# Build network
network = nx.DiGraph()

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse("airlines.xml")
collection = DOMTree.documentElement

# Add nodes
nodes = collection.getElementsByTagName("node")
for node in nodes:
    key = int(node.getAttribute("id"))
    args = {}
    for data in node.getElementsByTagName("data"):
        args[data.getAttribute("key")] = data.childNodes.item(0).data
    network.add_node(key, attr=args)

# Add edges
edges = collection.getElementsByTagName("edge")
for edge in edges:
    key = int(edge.getAttribute("id"))
    source = int(edge.getAttribute("source"))
    target = int(edge.getAttribute("target"))
    #count = edge.getElementsByTagName("data")[0].childNodes.item(0).data
    network.add_edge(source, target, old_key=key)

# Build nodes
data_nodes = []
for node in network.nodes(data=True):
    data_nodes.append(ForceBundleFunctionalNumba.Point(float(node[1]['attr']['x']), float(node[1]['attr']['y']) * -1))
    #data_nodes.append((float(node[1]['attr']['x']), float(node[1]['attr']['y']) * -1))
#data_nodes = np.array(data_nodes, dtype=[('x', 'f4'), ('y', 'f4')])

# Build edges
data_edges = []
count = 0
limit = 5000
for source, target in network.edges:
    #data_edges.append((source, target))
    if count >= limit:
        break
    count += 1
    data_edges.append(ForceBundleFunctionalNumba.Edge(data_nodes[source], data_nodes[target]))

# Set settings
eps = 1e-6
compatibility_threshold = 0.6

#edge_length = partial(ForceBundleFunctionalNumba.edge_length, eps)

#angle_compatibility = partial(ForceBundleFunctionalNumba.angle_compatibility)

#angle_compatibility(data_edges[69], data_edges[66])
#scale_compatibility = partial(ForceBundleFunctionalNumba.scale_compatibility)

#position_compatibility = partial(ForceBundleFunctionalNumba.position_compatibility)

#are_compatible = partial(ForceBundleFunctionalNumba.are_compatible, compatibility_threshold)


data = ForceBundleFunctionalNumba.compute_compatibility_list(data_edges)