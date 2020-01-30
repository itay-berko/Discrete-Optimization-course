#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import copy
import numpy
import pandas as pd
import numpy as np
import random

# i will start the search with different seq each time.
# in opt1 i will start the search with the longest edge.


Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def initial_solution(points,nodeCount, dis_matrix, first_point):
    search_list = []
    for i in range(nodeCount):
        search_list.append(i)

    initial_seq = list()
    initial_seq.append(search_list[first_point])
    initial_seq_index = 0
    del search_list[first_point]

    while True:
        # if they have same length --> i add all the point the the initial_seq
        if len(initial_seq) == nodeCount:
            break
        # initial the min distance for the curr item in initial_seq
        if search_list[0] == 0:
            min_val = dis_matrix[initial_seq[initial_seq_index]][search_list[1]]
        else:
            min_val = dis_matrix[initial_seq[initial_seq_index]][search_list[0]]
        # starting to search the real min value
        search_list_index = 0
        for i in range(len(search_list)):
            dis = dis_matrix[initial_seq[initial_seq_index]][search_list[i]]

            if dis == 0:
                continue

            if dis < min_val:
                min_val = dis
                search_list_index = i

        initial_seq.append(search_list[search_list_index])
        initial_seq_index += 1
        del search_list[search_list_index]

    return initial_seq


def create_distance_matrix(points):
    dis_matrix = numpy.zeros((len(points), len(points)))

    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            dis = length(points[i], points[j])
            dis_matrix[i][j] = dis
            dis_matrix[j][i] = dis

    return dis_matrix


# return the point with the max distance to the next point
def get_max_edge(dis_matrix, curr_seq):
    # initial the the variable to the last point in the seq
    max_index = len(curr_seq) - 1
    max_dis = dis_matrix[curr_seq[max_index]][curr_seq[0]]

    for i in range(0, len(curr_seq) - 1):
        if max_dis < dis_matrix[curr_seq[i]][curr_seq[i + 1]]:
            max_index = i
            max_dis = dis_matrix[curr_seq[i]][curr_seq[i + 1]]

    return max_index, max_dis


def get_shorter_edge(dis_matrix, curr_seq, origin_index, next_origin_old_index, max_dis):
    row = dis_matrix[curr_seq[next_origin_old_index]]

    # finding the next point in the curr_seq for next_origin_old_index
    if next_origin_old_index == len(curr_seq) - 1:
        next_point = curr_seq[0]
    else:
        next_point = curr_seq[next_origin_old_index + 1]

    close_point = None
    # finding the number of the point with length less then max_dis
    for i in range(len(row)):
        # we don't want the chose edge that connect old-->next in seq or old-->old or old-->origin
        if row[i] == 0 or i == curr_seq[origin_index] or i == next_point:
            continue
        elif row[i] < max_dis:
            close_point = i
            break

    # finding the index of the close_point in curr_seq
    if close_point is None:
        return None
    else:
        for i in range(len(curr_seq)):
            if curr_seq[i] == close_point:
                # return the prev point the close_point
                if i == 0:
                    return len(curr_seq) - 1
                else:
                    return i - 1

    return None


def modify_sequence(curr_seq, old_point_index, new_point_index):
    # the unchanged part of the sequence list:
    new_seq = []
    #  index in new_seq
    index = -1

    if new_point_index < old_point_index:

        for i in range(len(curr_seq) - 1, old_point_index - 1, -1):
            new_seq.append(curr_seq[i])
            index += 1

        for i in range(new_point_index + 1, old_point_index):
            new_seq.append(curr_seq[i])
            index += 1

        origin_index = index
        next_point_index = origin_index + 1

        for i in range(new_point_index, -1, -1):
            new_seq.append(curr_seq[i])

    else:

        for i in range(0, old_point_index):
            new_seq.append(curr_seq[i])
            index += 1

        origin_index = index
        next_point_index = origin_index + 1

        for i in range(new_point_index, old_point_index - 1, -1):
            new_seq.append(curr_seq[i])

        for i in range(new_point_index + 1, len(curr_seq)):
            new_seq.append(curr_seq[i])

    return new_seq, origin_index, next_point_index


def get_the_closest_points(curr_seq, dis_matrix, origin_index):
    row = dis_matrix[curr_seq[origin_index]]
    # initial the points
    if curr_seq[origin_index] == 0 or curr_seq[origin_index] == 1:
        point1 = 2
        dis_point1 = row[2]
        point2 = 3
        dis_point2 = row[3]
    else:
        point1 = 0
        dis_point1 = row[0]
        point2 = 1
        dis_point2 = row[1]
    # point 1 will have the min length
    if dis_point1 > dis_point2:
        point1, dis_point1, point2, dis_point2 = switch_point(point1, dis_point1, point2, dis_point2)

    for i in range(len(row)):
        if i == curr_seq[origin_index]:
            continue
        elif row[i] < dis_point1:
            point1, dis_point1, point2, dis_point2 = switch_point(i, row[i], point1, dis_point1)
            continue
        elif row[i] < dis_point2:
            point2 = i
            dis_point2 = row[i]

    return point1, point2


def calc_fun_value(seq, dis_matrix):
    value = 0

    for i in range(0, len(seq) - 1):
        curr_p = seq[i]
        next_p = seq[i + 1]
        value += dis_matrix[curr_p][next_p]

    value += dis_matrix[len(seq) - 1][seq[0]]

    return value


# find the opt k in the df
def find_min_value(df):
    # initial the variables to the first row
    min_val = df.loc[0][0]
    k = 0

    for i in range(len(df)):
        if min_val > df.loc[i][0]:
            min_val = df.loc[i][0]
            k = i
    return k


def get_index(point, seq):
    for i in range(len(seq)):
        if seq[i] == point:
            return i

    return None


def create_opt_seq(df, original_seq, start_origin_index, k_index):
    curr_seq = original_seq.copy()

    for i in range(0, k_index + 1):
        old_point = df.loc[i][1][0]
        old_point_index = get_index(old_point, curr_seq)
        new_point = df.loc[i][1][1]
        new_point_index = get_index(new_point, curr_seq)
        curr_seq = modify_sequence(curr_seq, old_point_index, new_point_index)[0]

    return curr_seq


def K_OPT(original_seq, dis_matrix, start_origin_index, edge_dis):
    df = pd.DataFrame(columns=['fun_val', 'switch points'])

    curr_seq = original_seq.copy()

    #  max number of iteration the find the k_opt
    num_iter = 10
    iter_counter = 0

    # the index of the point in curr_seq with the longest edge
    origin_index = start_origin_index
    max_dis = edge_dis

    # the point at the edge of the longest edge
    if origin_index < len(curr_seq) - 1:
        next_origin_old_index = origin_index + 1
    else:
        next_origin_old_index = 0

    while True:
        # finding shorter edge from the point 'next_origin_old_index'
        # and return the point that connect to the end of the edge
        next_origin_new_index = get_shorter_edge(dis_matrix, curr_seq, origin_index, next_origin_old_index, max_dis)

        if iter_counter >= num_iter or next_origin_new_index is None:
            break

        new_seq, new_origin_index, next_point_index = modify_sequence(curr_seq, next_origin_old_index,
                                                                      next_origin_new_index)

        fun_value = calc_fun_value(new_seq, dis_matrix)

        df = df.append(
            {'fun_val': fun_value, 'switch points': [curr_seq[next_origin_old_index], curr_seq[next_origin_new_index]]},
            ignore_index=True)

        # update the variable to the next loop
        curr_seq = new_seq
        origin_index = new_origin_index
        next_origin_old_index = next_point_index
        max_dis = dis_matrix[curr_seq[origin_index]][curr_seq[next_origin_old_index]]
        iter_counter += 1

    # modify the seq according the k opt, k is the index of the row in the df
    if len(df) > 0:
        k_index = find_min_value(df)
        # create the current optimal seq
        opt_seq = create_opt_seq(df, original_seq, start_origin_index, k_index)
        return opt_seq
    else:
        return None


def update_index(point, seq):
    for i in range(len(seq)):
        if seq[i] == point:
            return i

    return None


def the_best_neighbor(neighbors):
    min_value = neighbors.loc[0][2]
    neighbor = neighbors.loc[0]

    for i in range(len(neighbors)):
        if neighbors.loc[i][2] < min_value:
            min_value = neighbors.loc[i][2]
            neighbor = neighbors.loc[i]

    return neighbor


# heuristics search best neighbors
def local_search_opt1(dis_matrix, initial_sequence):
    # start improve the  seq with the longest edge
    start_origin_index, max_dis = get_max_edge(dis_matrix, initial_sequence)
    curr_opt_seq = K_OPT(initial_sequence, dis_matrix, start_origin_index, max_dis)
    # update the start_origin_index to the index in curr_opt_seq
    start_origin_index = update_index(initial_sequence[start_origin_index], curr_opt_seq)
    initial_fun_val = calc_fun_value(curr_opt_seq, dis_matrix)

    max_iter = 100
    iter_counter = 0
    num_neighbors = 20
    first_neighbors = pd.DataFrame(columns=['index', 'seq', 'fun_val'])
    neighbors_count = 0

    while iter_counter < max_iter:

        for i in range(len(curr_opt_seq)):
            # if i find enough neighbors I will stop the for loop
            if neighbors_count == num_neighbors:
                break
            # i don't want the check the current origin again
            if i == start_origin_index:
                continue

            if i == len(curr_opt_seq) - 1:
                edge_dis = dis_matrix[curr_opt_seq[i]][curr_opt_seq[0]]
            else:
                edge_dis = dis_matrix[curr_opt_seq[i]][curr_opt_seq[i + 1]]

            curr_seq = K_OPT(curr_opt_seq, dis_matrix, i, edge_dis)
            if curr_seq is None:
                continue

            curr_fun_val = calc_fun_value(curr_seq, dis_matrix)

            if curr_fun_val < initial_fun_val:
                first_neighbors = first_neighbors.append({'index': i, 'seq': curr_seq, 'fun_val': curr_fun_val},
                                                         ignore_index=True)
                neighbors_count += 1

        # initial new loop
        # select randomly one of the neighbors
        num_of_actual_neighbors = len(first_neighbors)
        if num_of_actual_neighbors == 0:
            break

        if num_of_actual_neighbors == 1:
            neighbor = first_neighbors.loc[0]
        else:
            # neighbor = the_best_neighbor(first_neighbors)
            neighbor = first_neighbors.loc[random.randrange(num_of_actual_neighbors - 1)]

        start_origin_index = update_index(curr_opt_seq[neighbor[0]], neighbor[1])
        curr_opt_seq = neighbor[1]
        initial_fun_val = neighbor[2]
        print("the new val in opt 1: ", initial_fun_val)

        first_neighbors = pd.DataFrame(columns=['index', 'seq', 'fun_val'])
        neighbors_count = 0

        iter_counter += 1

    return curr_opt_seq


# random walk
def local_search_opt2(dis_matrix, initial_sequence, not_visited):
    # initial randomly
    if len(not_visited) == 1:
        index_in_visited = 0
    else:
        index_in_visited = random.randrange(len(not_visited) - 1)

    index_in_initial = update_index(not_visited[index_in_visited], initial_sequence)

    del not_visited[index_in_visited]

    if index_in_initial == len(initial_sequence) - 1:
        edge_dis = dis_matrix[initial_sequence[index_in_initial]][initial_sequence[0]]
    else:
        edge_dis = dis_matrix[initial_sequence[index_in_initial]][initial_sequence[index_in_initial + 1]]

    opt_seq = K_OPT(initial_sequence, dis_matrix, index_in_initial, edge_dis)
    if opt_seq is None:
        opt_seq = initial_sequence
        min_fun_val = calc_fun_value(opt_seq, dis_matrix)
    else:
        min_fun_val = calc_fun_value(opt_seq, dis_matrix)

    # update the start_origin_index to the index in curr_opt_seq
    start_origin_index = update_index(initial_sequence[index_in_initial], opt_seq)
    min_fun_val = calc_fun_value(opt_seq, dis_matrix)

    max_iter = 2
    iter_counter = 0

    while iter_counter < max_iter:

        i = random.randrange(len(opt_seq) - 1)
        # i don't want the chose the same index
        while i == start_origin_index:
            i = random.randrange(len(opt_seq) - 1)

        if i == len(opt_seq) - 1:
            edge_dis = dis_matrix[opt_seq[i]][opt_seq[0]]
        else:
            edge_dis = dis_matrix[opt_seq[i]][opt_seq[i + 1]]

        curr_seq = K_OPT(opt_seq, dis_matrix, i, edge_dis)

        if curr_seq is None:
            continue

        curr_fun_val = calc_fun_value(curr_seq, dis_matrix)

        if curr_fun_val < min_fun_val:
            # update the variable
            start_origin_index = update_index(opt_seq[i], curr_seq)
            opt_seq = curr_seq
            min_fun_val = curr_fun_val

        iter_counter += 1

    return opt_seq


# creating new point that i didnwt visited and add her to the visited list
def create_new_point(visited, nodeCount):

    first_point = random.randrange(nodeCount - 1)

    while True:

        in_list = False

        for i in visited:
            if i == first_point:
                in_list = True
                break
        if in_list:
            first_point = random.randrange(nodeCount - 1)
        else:
            break

    visited.append(first_point)
    return first_point


def multi_local_search(points, dis_matrix, nodeCount):
    # list of all the points that I visited
    visited = list()
    # initial solution
    first_point = create_new_point(visited, nodeCount)
    initial_sequence = initial_solution(points, nodeCount, dis_matrix, first_point)
    # opt the initial solution
    opt_seq = local_search_opt1(dis_matrix, initial_sequence)

    if opt_seq is None:
        opt_seq = initial_sequence
        min_fun_val = calc_fun_value(opt_seq, dis_matrix)
    else:
        min_fun_val = calc_fun_value(opt_seq, dis_matrix)

    iteration = 200

    for i in range(iteration):
        print("start new search")
        first_point = create_new_point(visited, nodeCount)
        initial_sequence = initial_solution(points, nodeCount, dis_matrix, first_point)
        curr_seq = local_search_opt1(dis_matrix, initial_sequence)
        if curr_seq is None:
            continue
        fun_val = calc_fun_value(curr_seq, dis_matrix)
        if fun_val < min_fun_val:
            print("in multi search updating the val to =",fun_val )
            min_fun_val = fun_val
            opt_seq = curr_seq

    return opt_seq


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # create initial by connecting the point according to the order of points list
    dis_matrix = create_distance_matrix(points)
    opt_seq = multi_local_search(points, dis_matrix, nodeCount)

    solution = opt_seq

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    import sys

    file_location = r"data/tsp_76_1"
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))
