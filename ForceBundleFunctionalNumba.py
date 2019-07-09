import numpy as np
from numba import jitclass, float32, jit, njit
from numba.types import pyobject, uint8
from functools import partial
from tqdm.auto import tqdm
import math


EPS = 1e-6
#self.compatibility_threshold = 0.6
COMPATIBILITY_THRESHOLD = 0.6

@jitclass([('x', float32), ('y', float32)])
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(x={self.x!r}, y={self.y!r})')

@jitclass([
    ('source', Point.class_type.instance_type),
    ('target', Point.class_type.instance_type),
])
class Edge:
    def __init__(self, source, target):
        self.source = source
        self.target = target

#@profile
@jit(Point.class_type.instance_type(Edge.class_type.instance_type), parallel=True, nopython=True, fastmath=True)
def edge_as_vector(edge):
    return Point(edge.target.x - edge.source.x, edge.target.y - edge.source.y)

#@profile
@jit(float32(Edge.class_type.instance_type), nopython=True, parallel=True, fastmath=True)
def edge_length(edge):
    # handling nodes that are the same location, so that K / edge_length != Inf
    if (abs(edge.source.x - edge.target.x)) < EPS and \
            (abs(edge.source.y - edge.target.y)) < EPS:
        return EPS

    return np.sqrt(np.power(edge.source.x - edge.target.x, 2) +
                     np.power(edge.source.y - edge.target.y, 2))

#@profile
@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type), nopython=True, fastmath=True)
def angle_compatibility(edge, oedge):
    v1 = edge_as_vector(edge)
    v2 = edge_as_vector(oedge)
    #quotient = np.dot(v1, v2)
    quotient = v1.x * v2.x + v1.y * v2.y
    dividend = edge_length(edge) * edge_length(oedge)
    return abs(quotient / dividend)

#@profile
@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type), nopython=True, fastmath=True)
def scale_compatibility(edge, oedge):
    lavg = (edge_length(edge) + edge_length(oedge)) / 2.0
    return 2.0 / (lavg / min(edge_length(edge),
                             edge_length(oedge)) + max(edge_length(edge),
                                                          edge_length(oedge)) / lavg)

#@profile
@jit(float32(Point.class_type.instance_type, Point.class_type.instance_type), nopython=True, parallel=True, fastmath=True)
def euclidean_distance(source, target):
    return np.sqrt(np.power((source.x - target.x), 2) + np.power((source.y - target.y), 2))

#@profile
@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type), nopython=True, parallel=True, fastmath=True)
def position_compatibility(edge, oedge):
        lavg = (edge_length(edge) + edge_length(oedge)) / 2.0
        #print(lavg)
        midP = Point((edge.source.x + edge.target.x) / 2.0,
                (edge.source.y + edge.target.y) / 2.0)
        #print(midP)
        midQ = Point((oedge.source.x + oedge.target.x) / 2.0,
                     (oedge.source.y + oedge.target.y) / 2.0)
        #print(midQ)

        return lavg / (lavg + euclidean_distance(midP, midQ))

@jit(Point.class_type.instance_type(Point.class_type.instance_type, Edge.class_type.instance_type), nopython=True, parallel=True, fastmath=True)
def project_point_on_line(point, edge):
    L = math.sqrt((edge.target.x - edge.source.x) ** 2 +
                  (edge.target.y - edge.source.y) ** 2)
    r = ((edge.source.y - point.y) * (edge.source.y - edge.target.y) -
         (edge.source.x - point.x) * (edge.target.x - edge.source.x)) / \
        math.sqrt(L)
    #print(r)
    return Point((edge.source.x + r * (edge.target.x - edge.source.x)),
                 (edge.source.y + r * (edge.target.y - edge.source.y)))

@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type))
def edge_visibility(edge, oedge):
    # send actual edge points positions
    I0 = project_point_on_line(oedge.source, edge)
    I1 = project_point_on_line(oedge.target, edge)
    midI = Point((I0.x + I1.x) / 2.0, (I0.y + I1.y) / 2.0)

    midP = Point((edge.source.x + edge.target.x) / 2.0,
                 (edge.source.y + edge.target.y) / 2.0)

    #print(1 - 2 * euclidean_distance(np.array([midP, midI])) / euclidean_distance(np.array([I0, I1])))
    return max(0, 1 - 2 * euclidean_distance(midP, midI) / euclidean_distance(I0, I1))

@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type))
def visibility_compatibility(edge, oedge):
    return min(edge_visibility(edge, oedge), edge_visibility(oedge, edge))

#@profile
@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type))
def are_compatible(edge, oedge):
    #print((edge, oedge))
    angles_score = angle_compatibility(edge, oedge)
    scales_score = scale_compatibility(edge, oedge)
    positi_score = position_compatibility(edge, oedge)
    visivi_score = visibility_compatibility(edge, oedge)
    score = (angles_score * scales_score * positi_score * visivi_score)

    return score >= COMPATIBILITY_THRESHOLD

@profile
def compute_compatibility_list(data_edges):
    compatibility_list = []
    total_edges = len(data_edges)
    for e_idx in range(total_edges - 1):
        for oe_idx in range(e_idx, total_edges):
            if are_compatible(data_edges[e_idx], data_edges[oe_idx]):
                compatibility_list.append((data_edges[e_idx], data_edges[oe_idx]))
                compatibility_list.append((data_edges[oe_idx], data_edges[e_idx]))
    return compatibility_list