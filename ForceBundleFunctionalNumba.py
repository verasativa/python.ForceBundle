import numpy as np
from numba import jitclass, float32, jit, njit, prange
from numba.typed import Dict, List
from numba.types import ListType, uint8, int16
from functools import partial
from tqdm.auto import tqdm
import math, sys


EPS = 1e-6
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

ForceFactors = Point

@jit(Point.class_type.instance_type(Edge.class_type.instance_type), nopython=True, fastmath=True)
def edge_as_vector(edge):
    return Point(edge.target.x - edge.source.x, edge.target.y - edge.source.y)


@jit(float32(Edge.class_type.instance_type), nopython=True, fastmath=True)
def edge_length(edge):
    # handling nodes that are the same location, so that K / edge_length != Inf
    if (abs(edge.source.x - edge.target.x)) < EPS and (abs(edge.source.y - edge.target.y)) < EPS:
        return EPS

    return np.sqrt(np.power(edge.source.x - edge.target.x, 2) + np.power(edge.source.y - edge.target.y, 2))


@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type), nopython=True, fastmath=True)
def angle_compatibility(edge, oedge):
    v1 = edge_as_vector(edge)
    v2 = edge_as_vector(oedge)
    dot_product = v1.x * v2.x + v1.y * v2.y
    return math.fabs(dot_product / (edge_length(edge) * edge_length(oedge)))

#@profile
@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type), nopython=True, fastmath=True)
def scale_compatibility(edge, oedge):
    lavg = (edge_length(edge) + edge_length(oedge)) / 2.0
    return 2.0 / (lavg/min(edge_length(edge), edge_length(oedge)) + max(edge_length(edge), edge_length(oedge))/lavg)

#@profile
@jit(float32(Point.class_type.instance_type, Point.class_type.instance_type), nopython=True, fastmath=True)
def euclidean_distance(source, target):
    if source.x == target.x and source.y == target.y:
        return EPS
    return np.sqrt(np.power(source.x - target.x, 2) + np.power(source.y - target.y, 2))

#@profile
@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type), nopython=True, fastmath=True, cache=True)
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

@jit(Point.class_type.instance_type(Point.class_type.instance_type, Edge.class_type.instance_type), nopython=True, fastmath=True)
def project_point_on_line(point, edge):
    L = math.sqrt(np.power(edge.target.x - edge.source.x, 2) + np.power((edge.target.y - edge.source.y), 2))
    if L < EPS:
        L = EPS
    r = ((edge.source.y - point.y) * (edge.source.y - edge.target.y) - (edge.source.x - point.x) * (edge.target.x - edge.source.x)) / np.power(L, 2)
    #print(r)
    return Point((edge.source.x + r * (edge.target.x - edge.source.x)),
                 (edge.source.y + r * (edge.target.y - edge.source.y)))

@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type), nopython=True, fastmath=True)
def edge_visibility(edge, oedge):
    # send actual edge points positions
    I0 = project_point_on_line(oedge.source, edge)
    I1 = project_point_on_line(oedge.target, edge)
    midI = Point((I0.x + I1.x) / 2.0, (I0.y + I1.y) / 2.0)

    midP = Point((edge.source.x + edge.target.x) / 2.0,
                 (edge.source.y + edge.target.y) / 2.0)

    #print(1 - 2 * euclidean_distance(np.array([midP, midI])) / euclidean_distance(np.array([I0, I1])))
    return max(0, 1 - 2 * euclidean_distance(midP, midI) / euclidean_distance(I0, I1))


@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type), nopython=True, fastmath=True)
def visibility_compatibility(edge, oedge):
    # TODO: Implement both directions at edge_visibility
    return min(edge_visibility(edge, oedge), edge_visibility(oedge, edge))


@jit(float32(Edge.class_type.instance_type, Edge.class_type.instance_type), nopython=True, fastmath=True)
def are_compatible(edge, oedge):
    angles_score = angle_compatibility(edge, oedge)
    scales_score = scale_compatibility(edge, oedge)
    positi_score = position_compatibility(edge, oedge)
    visivi_score = visibility_compatibility(edge, oedge)

    score = (angles_score * scales_score * positi_score * visivi_score)

    #msg = 'angle_compatibility: {}\nscale_compatibility: {}\nposition_compatibility: {}\nvisibility_compatibility: {}\nTotal: {}\n'
    #print(msg.format(angles_score, scales_score, positi_score, visivi_score, score))
    #sys.exit()
    return score >= COMPATIBILITY_THRESHOLD

#@jit(ListType(ListType(int16))(ListType(Edge.class_type.instance_type)), nopython=True)
def compute_compatibility_list(data_edges):
    compatibility_list = List()
    for _ in data_edges:
        compatibility_list.append(List.empty_list(int16))
    total_edges = len(data_edges)
    for e_idx in tqdm(range(total_edges - 1), unit='Edges'):
        compatibility_list = compute_compatibility_list_on_edge(data_edges, e_idx, compatibility_list, total_edges)
#        for oe_idx in range(e_idx, total_edges):
#            if are_compatible(data_edges[e_idx], data_edges[oe_idx]):
#                compatibility_list[e_idx].append(oe_idx)
#                compatibility_list[oe_idx].append(e_idx)
    return compatibility_list

@jit(ListType(ListType(int16))(ListType(Edge.class_type.instance_type), int16, ListType(ListType(int16)), int16), nopython=True)
#@jit(nopython=True)
def compute_compatibility_list_on_edge(data_edges, e_idx, compatibility_list, total_edges):
    for oe_idx in range(e_idx, total_edges):
        if are_compatible(data_edges[e_idx], data_edges[oe_idx]):
            compatibility_list[e_idx].append(oe_idx)
            compatibility_list[oe_idx].append(e_idx)
    return compatibility_list

# Need to set types on var (they are not availables inside a jit function)
pt_cls = Point.class_type.instance_type
list_of_pts = ListType(pt_cls)
#@jit(ListType(ListType(Point.class_type.instance_type))(ListType(Edge.class_type.instance_type), uint8), nopython=True)
def build_edge_subdivisions(edges, P_initial=1):
    subdivision_points_for_edge = List.empty_list(list_of_pts)

    for i in range(len(edges)):
        subdivision_points_for_edge.append(List.empty_list(pt_cls))

        if P_initial != 1:
            subdivision_points_for_edge[i].append(edges[i].source)
            subdivision_points_for_edge[i].append(edges[i].target)

    return subdivision_points_for_edge


@jit(nopython=True, fastmath=True)
def compute_divided_edge_length(subdivision_points_for_edge, edge_idx):
    length = 0

    for i in range(1, len(subdivision_points_for_edge[edge_idx])):
        segment_length = euclidean_distance(subdivision_points_for_edge[edge_idx][i],
                                            subdivision_points_for_edge[edge_idx][i - 1])
        length += segment_length

    return length


@jit(Point.class_type.instance_type(Edge.class_type.instance_type), nopython=True, fastmath=True)
def edge_midpoint(edge):
    middle_x = (edge.source.x + edge.target.x) / 2
    middle_y = (edge.source.y + edge.target.y) / 2

    return Point(middle_x, middle_y)


@jit(nopython=True, fastmath=True)
def update_edge_divisions(data_edges, subdivision_points_for_edge, P):
    for edge_idx in range(len(data_edges)):
        if P == 1:
            subdivision_points_for_edge[edge_idx].append(data_edges[edge_idx].source)
            subdivision_points_for_edge[edge_idx].append(edge_midpoint(data_edges[edge_idx]))  # Pendiente
            subdivision_points_for_edge[edge_idx].append(data_edges[edge_idx].target)  # Pendiente
        else:
            divided_edge_length = compute_divided_edge_length(subdivision_points_for_edge, edge_idx)
            segment_length = divided_edge_length / (P + 1)
            current_segment_length = segment_length #.copy()
            new_subdivision_points = List()
            new_subdivision_points.append(data_edges[edge_idx].source)  # source

            for i in range(1, len(subdivision_points_for_edge[edge_idx])):
                old_segment_length = euclidean_distance(subdivision_points_for_edge[edge_idx][i],
                                                        subdivision_points_for_edge[edge_idx][i - 1])

                while (old_segment_length > current_segment_length):
                    percent_position = current_segment_length / old_segment_length
                    new_subdivision_point_x = subdivision_points_for_edge[edge_idx][i - 1].x
                    new_subdivision_point_y = subdivision_points_for_edge[edge_idx][i - 1].y

                    new_subdivision_point_x += percent_position * (
                                subdivision_points_for_edge[edge_idx][i].x - subdivision_points_for_edge[edge_idx][
                            i - 1].x)
                    new_subdivision_point_y += percent_position * (
                                subdivision_points_for_edge[edge_idx][i].y - subdivision_points_for_edge[edge_idx][
                            i - 1].y)

                    new_subdivision_points.append(Point(new_subdivision_point_x, new_subdivision_point_y))

                    old_segment_length -= current_segment_length
                    current_segment_length = segment_length

                current_segment_length -= old_segment_length

            new_subdivision_points.append(data_edges[edge_idx].target)  # target
            subdivision_points_for_edge[edge_idx] = new_subdivision_points

    return subdivision_points_for_edge


@jit(nopython=True, fastmath=True)
def apply_spring_force(subdivision_points_for_edge, edge_idx, i, kP):
    prev = subdivision_points_for_edge[edge_idx][i - 1]
    next_ = subdivision_points_for_edge[edge_idx][i + 1]
    crnt = subdivision_points_for_edge[edge_idx][i]
    x = prev.x - crnt.x + next_.x - crnt.x
    y = prev.y - crnt.y + next_.y - crnt.y

    x *= kP
    y *= kP

    return ForceFactors(x, y)

@jit(nopython=True, fastmath=True)
def custom_edge_length(edge):
    return math.sqrt(math.pow(edge.source.x - edge.target.x, 2) + math.pow(edge.source.y - edge.target.y, 2))

@jit(ForceFactors.class_type.instance_type(ListType(ListType(Point.class_type.instance_type)), ListType(ListType(int16)), int16, int16, ListType(float32)), nopython=True, fastmath=True) #
def apply_electrostatic_force(subdivision_points_for_edge, compatibility_list_for_edge, edge_idx, i, weights):
    #sum_of_forces = ForceFactors(0.0, 0.0)
    sum_of_forces_x = 0.0
    sum_of_forces_y = 0.0
    compatible_edges_list = compatibility_list_for_edge[edge_idx]
    use_weights = True if len(weights) > 0 else False

    for oe in range(len(compatible_edges_list)):
        if use_weights:
            force = ForceFactors((subdivision_points_for_edge[compatible_edges_list[oe]][i].x - subdivision_points_for_edge[edge_idx][i].x)  * weights[oe],
                                 (subdivision_points_for_edge[compatible_edges_list[oe]][i].y - subdivision_points_for_edge[edge_idx][i].y)  * weights[oe]
                                 )
        else:
            force = ForceFactors((subdivision_points_for_edge[compatible_edges_list[oe]][i].x - subdivision_points_for_edge[edge_idx][i].x),
                                 (subdivision_points_for_edge[compatible_edges_list[oe]][i].y - subdivision_points_for_edge[edge_idx][i].y)
                                 )


        if ((math.fabs(force.x) > EPS) or (math.fabs(force.y) > EPS)):
            diff = (1 / math.pow(custom_edge_length(Edge(subdivision_points_for_edge[compatible_edges_list[oe]][i],
                                                         subdivision_points_for_edge[edge_idx][i])), 1))

            #sum_of_forces = ForceFactors((force.x * diff) + sum_of_forces.x, (force.y * diff) + sum_of_forces.y)
            sum_of_forces_x += force.x * diff
            sum_of_forces_y += force.y * diff

    #return sum_of_forces
    return ForceFactors(sum_of_forces_x, sum_of_forces_y)

@jit(nopython=True, fastmath=True)
def apply_resulting_forces_on_subdivision_points(data_edges, subdivision_points_for_edge, compatibility_list_for_edge, edge_idx, K, P, S, weights):
    # kP = K / | P | (number of segments), where | P | is the initial length of edge P.
    kP = K / (edge_length(data_edges[edge_idx]) * (P + 1))

    # (length * (num of sub division pts - 1))
    resulting_forces_for_subdivision_points = [ForceFactors(0.0, 0.0)]

    for i in range(1, P + 1):
        spring_force = apply_spring_force(subdivision_points_for_edge, edge_idx, i, kP)
        electrostatic_force = apply_electrostatic_force(subdivision_points_for_edge, compatibility_list_for_edge, edge_idx, i, weights)

        resulting_force = ForceFactors(S * (spring_force.x + electrostatic_force.x),
                                       S * (spring_force.y + electrostatic_force.y))

        resulting_forces_for_subdivision_points.append(resulting_force)


    resulting_forces_for_subdivision_points.append(ForceFactors(0.0, 0.0))

    return resulting_forces_for_subdivision_points

# No numba, so we have tqdm
def forcebundle(edges, S_initial, I_initial, I_rate, P_initial, P_rate, C, K, weights):
    S = S_initial
    I = I_initial
    P = P_initial

    subdivision_points_for_edge = build_edge_subdivisions(edges, P_initial)
    compatibility_list_for_edge = compute_compatibility_list(edges)
    subdivision_points_for_edge = update_edge_divisions(edges, subdivision_points_for_edge, P)

    for _cycle in tqdm(range(C), unit='cycle'):
        subdivision_points_for_edge, S, P, I = apply_forces_cycle(edges, subdivision_points_for_edge, compatibility_list_for_edge, K, P, P_rate, I, I_rate, S, weights)

    return subdivision_points_for_edge

@jit(nopython=True)
def apply_forces_cycle(edges, subdivision_points_for_edge, compatibility_list_for_edge, K, P, P_rate, I, I_rate, S, weights):
    for iteration in range(math.ceil(I)):
        forces = List()
        for edge_idx in range(len(edges)):
            forces.append(apply_resulting_forces_on_subdivision_points(edges, subdivision_points_for_edge,
                                                                            compatibility_list_for_edge, edge_idx, K, P,
                                                                            S, weights))
            for i in range(P):
                subdivision_points_for_edge[edge_idx][i] = Point(
                    subdivision_points_for_edge[edge_idx][i].x + forces[edge_idx][i].x,
                    subdivision_points_for_edge[edge_idx][i].y + forces[edge_idx][i].y
                )

    # prepare for next cycle
    S = S / 2
    P = P * P_rate
    I = I * I_rate

    subdivision_points_for_edge = update_edge_divisions(edges, subdivision_points_for_edge, P)
    #print('C: {} P: {} S: {}'.format(cycle, P, S))

    return subdivision_points_for_edge, S, P, I