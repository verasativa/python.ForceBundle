import ForceBundleFunctionalNumba
import networkx as nx
from xml.dom.minidom import parse
import xml.dom.minidom
from numba.typed import List

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
data_edges = List()
count = 0
limit = 10
for source, target in network.edges:
    #data_edges.append((source, target))
    if count >= limit:
        break
    count += 1
    data_edges.append(ForceBundleFunctionalNumba.Edge(data_nodes[source], data_nodes[target]))

# Set hyper-parameters:
#
# global bundling constant controlling edge stiffness
K = 0.1
# init.distance to move points
S_initial = 0.1
# init. subdivision number
P_initial = 1
# subdivision rate increase
P_rate = 2
# number of cycles to perform
C = 6
# init.number of iterations for cycle
I_initial = 90
# rate at which iteration number decreases i.e. 2 / 3
I_rate = 0.6666667

output = ForceBundleFunctionalNumba.forcebundle(data_edges, S_initial, I_initial, I_rate, P_initial, P_rate, C, K)