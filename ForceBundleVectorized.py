import math
#from collections import namedtuple
from tqdm.auto import tqdm
import numpy as np
from numba import jitclass, float32
from numba.types import pyobject, uint8

#Point = namedtuple('Point', 'x y')
#Edge = namedtuple('Edge', 'source target')
@jitclass([
    ('x', float32),
    ('y', float32),
])

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

@jitclass([
    ('source', Point.class_type.instance_type),
    ('target', Point.class_type.instance_type),
])

class Edge:
    def __init__(self, source, target):
        self.source = source
        self.target = target

ForceFactors = Point

@jitclass([
    ('data_nodes', pyobject),
    ('data_edges', pyobject),
    ('compatibility_list_for_edge', pyobject),
    ('subdivision_points_for_edge', pyobject),
    ('K', float32),
    ('S_initial', float32),
    ('P_rate', uint8),
    ('C', uint8),
    ('I_rate', float32),
    ('compatibility_threshold', float32),
    ('eps', float32),
])
class ForceEdgeBundling():
    def __init__(self, data_nodes, data_edges):
        self.data_nodes = data_nodes
        self.data_edges = data_edges

        self.compatibility_list_for_edge = {}
        self.subdivision_points_for_edge = {}

        # global bundling constant controlling edge stiffness
        self.K = 0.1
        # init.distance to move points
        self.S_initial = 0.1
        # init. subdivision number
        self.P_initial = 1
        # subdivision rate increase
        self.P_rate = 2
        # number of cycles to perform
        self.C = 6 #6
        # init.number of iterations for cycle
        self.I_initial = 90 #90
        # rate at which iteration number decreases i.e. 2 / 3
        self.I_rate = 0.6666667
        self.compatibility_threshold = 0.6
        self.eps = 1e-6

    def initialize_edge_subdivisions(self):
        for i in self.data_edges.keys():
            if self.P_initial == 1:
                self.subdivision_points_for_edge[i] = []
            else:
                self.subdivision_points_for_edge[i] = []
                self.subdivision_points_for_edge[i].append(self.data_nodes[self.data_edges[i].source])
                self.subdivision_points_for_edge[i].append(self.data_nodes[self.data_edges[i].target])

    def edge_midpoint(self, edge):
        middle_x = (self.data_nodes[edge.source].x + self.data_nodes[edge.target].x) / 2
        middle_y = (self.data_nodes[edge.source].y + self.data_nodes[edge.target].y) / 2

        return Point(middle_x, middle_y)

    def euclidean_distance(self, p, q):
        return math.sqrt(np.power(p.x - q.x, 2) + np.power(p.y - q.y, 2))

    def project_point_on_line(self, p, Q):
        L = math.sqrt((Q.target.x - Q.source.x) * (Q.target.x - Q.source.x) + (Q.target.y - Q.source.y) * (Q.target.y - Q.source.y))
        r = ((Q.source.y - p.y) * (Q.source.y - Q.target.y) - (Q.source.x - p.x) * (Q.target.x - Q.source.x)) / (L * L)

        return Point((Q.source.x + r * (Q.target.x - Q.source.x)),
                     (Q.source.y + r * (Q.target.y - Q.source.y)))

    def compute_divided_edge_length(self, edge_idx):
        length = 0

        for i in range(1, len(self.subdivision_points_for_edge[edge_idx])): # Maybe +1 or -1?
            segment_length = self.euclidean_distance(self.subdivision_points_for_edge[edge_idx][i],
                                                     self.subdivision_points_for_edge[edge_idx][i - 1])
            length += segment_length


        return length

    def update_edge_divisions(self, P):
        for edge_idx in self.data_edges.keys():
            if P == 1:
                self.subdivision_points_for_edge[edge_idx].append(self.data_nodes[self.data_edges[edge_idx].source])
                self.subdivision_points_for_edge[edge_idx].append(self.edge_midpoint(self.data_edges[edge_idx]))
                self.subdivision_points_for_edge[edge_idx].append(self.data_nodes[self.data_edges[edge_idx].target])
            else:
                divided_edge_length = self.compute_divided_edge_length(edge_idx)
                segment_length = divided_edge_length / (P + 1)
                current_segment_length = segment_length #.copy()
                new_subdivision_points = []
                new_subdivision_points.append(self.data_nodes[self.data_edges[edge_idx].source]) # source

                for i in range(1, len(self.subdivision_points_for_edge[edge_idx])):
                    old_segment_length = self.euclidean_distance(self.subdivision_points_for_edge[edge_idx][i],
                                                                 self.subdivision_points_for_edge[edge_idx][i - 1])
                    while (old_segment_length > current_segment_length):
                        percent_position = current_segment_length / old_segment_length
                        new_subdivision_point_x = self.subdivision_points_for_edge[edge_idx][i - 1].x
                        new_subdivision_point_y = self.subdivision_points_for_edge[edge_idx][i - 1].y

                        new_subdivision_point_x += percent_position * (
                                    self.subdivision_points_for_edge[edge_idx][i].x - self.subdivision_points_for_edge[edge_idx][
                                i - 1].x)
                        new_subdivision_point_y += percent_position * (
                                self.subdivision_points_for_edge[edge_idx][i].y - self.subdivision_points_for_edge[edge_idx][
                                i - 1].y)
                        new_subdivision_points.append(Point(new_subdivision_point_x, new_subdivision_point_y))

                        old_segment_length -= current_segment_length
                        current_segment_length = segment_length

                    current_segment_length -= old_segment_length

                new_subdivision_points.append(self.data_nodes[self.data_edges[edge_idx].target]) # target
                self.subdivision_points_for_edge[edge_idx] = new_subdivision_points

    def custom_edge_length(self, edge):
        return math.sqrt(math.pow(edge.source.x - edge.target.x, 2) + math.pow(edge.source.y - edge.target.y, 2))

    def apply_spring_force(self, edge_idx, i, kP):
        prev = self.subdivision_points_for_edge[edge_idx][i - 1]
        next = self.subdivision_points_for_edge[edge_idx][i + 1]
        crnt = self.subdivision_points_for_edge[edge_idx][i]
        x = prev.x - crnt.x + next.x - crnt.x
        y = prev.y - crnt.y + next.y - crnt.y

        x *= kP
        y *= kP

        return ForceFactors(x, y)

    def apply_electrostatic_force(self, edge_idx, i, S):
        #sum_of_forces = ForceFactors(0.0, 0.0)
        sum_of_forces_x = 0.0
        sum_of_forces_y = 0.0
        compatible_edges_list = self.compatibility_list_for_edge[edge_idx]

        for oe in range(0, len(compatible_edges_list)):
            force = ForceFactors(self.subdivision_points_for_edge[compatible_edges_list[oe]][i].x - self.subdivision_points_for_edge[edge_idx][i].x,
                                 self.subdivision_points_for_edge[compatible_edges_list[oe]][i].y - self.subdivision_points_for_edge[edge_idx][i].y
                                 )

            if ((math.fabs(force.x) > self.eps) or (math.fabs(force.y) > self.eps)):
                diff = (1 / math.pow(self.custom_edge_length(Edge(self.subdivision_points_for_edge[compatible_edges_list[oe]][i],
                                                                  self.subdivision_points_for_edge[edge_idx][i])), 1))

                #sum_of_forces = ForceFactors((force.x * diff) + sum_of_forces.x, (force.y * diff) + sum_of_forces.y)
                sum_of_forces_x += force.x * diff
                sum_of_forces_y += force.y * diff

        #return sum_of_forces
        return ForceFactors(sum_of_forces_x, sum_of_forces_y)

    def apply_resulting_forces_on_subdivision_points(self, edge_idx, P, S):
        # kP = K / | P | (number of segments), where | P | is the initial length of edge P.
        kP = self.K / (self.edge_length(self.data_edges[edge_idx]) * (P + 1))

        # (length * (num of sub division pts - 1))
        resulting_forces_for_subdivision_points = [ForceFactors(0.0, 0.0)]

        for i in range(1, P+ 1):
            resulting_force = ForceFactors(0.0, 0.0)
            spring_force = self.apply_spring_force(edge_idx, i, kP)
            electrostatic_force = self.apply_electrostatic_force(edge_idx, i, S)

            resulting_force = ForceFactors(S * (spring_force.x + electrostatic_force.x),
                                           S * (spring_force.y + electrostatic_force.y))
            #resulting_force.x = S * (spring_force.x + electrostatic_force.x)
            #resulting_force.y = S * (spring_force.y + electrostatic_force.y)

            resulting_forces_for_subdivision_points.append(resulting_force)


        resulting_forces_for_subdivision_points.append(ForceFactors(0.0, 0.0))

        return resulting_forces_for_subdivision_points

    def initialize_compatibility_lists(self):
        for i in self.data_edges.keys():
            self.compatibility_list_for_edge[i] = [] # 0 compatible edges.

    def compute_compatibility_lists(self):
        for e in tqdm(list(self.data_edges.keys())[:-1], unit='Edge (compatibility list)'):
            for oe in list(self.data_edges.keys())[e + 1:]:
                if self.are_compatible(self.data_edges[e], self.data_edges[oe]):
                    self.compatibility_list_for_edge[e].append(oe)
                    self.compatibility_list_for_edge[oe].append(e)
    #
    # Geometry helper methods
    #
    def vector_dot_product(self, p, q):
        return p.x * q.x + p.y * q.y

    def edge_as_vector(self, P):
        return Point(self.data_nodes[P.target].x - self.data_nodes[P.source].x,
                     self.data_nodes[P.target].y - self.data_nodes[P.source].y)

    def edge_length(self, edge):
        # handling nodes that are the same location, so that K / edge_length != Inf
        if (math.fabs(self.data_nodes[edge.source].x - self.data_nodes[edge.target].x) < self.eps and
                math.fabs(self.data_nodes[edge.source].y - self.data_nodes[edge.target].y) < self.eps):
            return self.eps

        return math.sqrt(math.pow(self.data_nodes[edge.source].x - self.data_nodes[edge.target].x, 2) +
                         math.pow(self.data_nodes[edge.source].y - self.data_nodes[edge.target].y, 2))

    #
    # Edge compatibility measures (functions)
    #
    def angle_compatibility(self, P, Q):
        return math.fabs(self.vector_dot_product(self.edge_as_vector(P), self.edge_as_vector(Q)) / (self.edge_length(P) * self.edge_length(Q)))

    def scale_compatibility(self, P, Q):
        lavg = (self.edge_length(P) + self.edge_length(Q)) / 2.0
        return 2.0 / (lavg / min(self.edge_length(P), self.edge_length(Q)) + max(self.edge_length(P), self.edge_length(Q)) / lavg)

    def position_compatibility(self, P, Q):
        lavg = (self.edge_length(P) + self.edge_length(Q)) / 2.0
        midP = Point((self.data_nodes[P.source].x + self.data_nodes[P.target].x) / 2.0,
                     (self.data_nodes[P.source].y + self.data_nodes[P.target].y) / 2.0)
        midQ = Point((self.data_nodes[Q.source].x + self.data_nodes[Q.target].x) / 2.0,
                     (self.data_nodes[Q.source].y + self.data_nodes[Q.target].y) / 2.0)

        return lavg / (lavg + self.euclidean_distance(midP, midQ))

    def edge_visibility(self, P, Q):
        # send actual edge points positions
        I0 = self.project_point_on_line(self.data_nodes[Q.source],
                                        Edge(self.data_nodes[P.source], self.data_nodes[P.target]))
        I1 = self.project_point_on_line(self.data_nodes[Q.target],
                                        Edge(self.data_nodes[P.source], self.data_nodes[P.target]))
        midI = Point((I0.x + I1.x) / 2.0, (I0.y + I1.y) / 2.0)

        midP = Point((self.data_nodes[P.source].x + self.data_nodes[P.target].x) / 2.0,
                     (self.data_nodes[P.source].y + self.data_nodes[P.target].y) / 2.0)

        return max(0, 1 - 2 * self.euclidean_distance(midP, midI) / self.euclidean_distance(I0, I1))

    def visibility_compatibility(self, P, Q):
        return min(self.edge_visibility(P, Q), self.edge_visibility(Q, P))

    def compatibility_score(self, P, Q):
        return (self.angle_compatibility(P, Q) * self.scale_compatibility(P, Q) *
                self.position_compatibility(P, Q) * self.visibility_compatibility(P, Q))

    def are_compatible(self, P, Q):
        return (self.compatibility_score(P, Q) >= self.compatibility_threshold)

    def forcebundle(self):
        S = self.S_initial
        I = self.I_initial
        P = self.P_initial

        self.initialize_edge_subdivisions()
        self.initialize_compatibility_lists()
        self.update_edge_divisions(P)
        self.compute_compatibility_lists()

        for cycle in tqdm(range(0, self.C), unit='cycle'):
            for iteration in range(math.ceil(I)):
                forces = {}
                for edge in self.data_edges.keys():
                    forces[edge] = self.apply_resulting_forces_on_subdivision_points(edge, P, S)
                for e in self.data_edges.keys():
                    for i in range(P):
                        self.subdivision_points_for_edge[e][i] = Point(
                            self.subdivision_points_for_edge[e][i].x + forces[e][i].x,
                            self.subdivision_points_for_edge[e][i].y + forces[e][i].y
                        )
                        #self.subdivision_points_for_edge[e][i].x += forces[e][i].x
                        #self.subdivision_points_for_edge[e][i].y += forces[e][i].y

            # prepare for next cycle
            S = S / 2
            P = P * self.P_rate
            I = I * self.I_rate

            self.update_edge_divisions(P)
            print('C: {} P: {} S: {}'.format(cycle, P, S))

        return self.subdivision_points_for_edge
