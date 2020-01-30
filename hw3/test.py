#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import copy
import numpy
import pandas as pd
import numpy as np
import random

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def initial_solution(points):

    initial_seq = []

    for i in range(0, len(points)):
        initial_seq.append(i)
    return initial_seq


def create_distance_matrix(points):

    dis_matrix = numpy.zeros((len(points),len(points)))

    for i in range(len(points) -1):
        for j in range(i+1,len(points)):

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
        if max_dis < dis_matrix[curr_seq[i]][curr_seq[i+1]]:
            max_index = i
            max_dis = dis_matrix[curr_seq[i]][curr_seq[i+1]]

    return max_index, max_dis


def get_shorter_edge(dis_matrix, curr_seq, origin_index, next_origin_old_index, distance):

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
        elif row[i] < distance:
            distance = row[i]
            close_point = i


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

        for i in range(0,old_point_index):
            new_seq.append(curr_seq[i])
            index += 1

        origin_index = index
        next_point_index = origin_index + 1

        for i in range(new_point_index, old_point_index - 1, -1):
            new_seq.append(curr_seq[i])

        for i in range (new_point_index + 1, len(curr_seq)):
            new_seq.append(curr_seq[i])

    return new_seq, origin_index, next_point_index


def calc_fun_value(seq, dis_matrix):

    value = 0

    for i in range(0, len(seq) - 1):
        curr_p= seq[i]
        next_p = seq[i+1]
        value +=dis_matrix[curr_p][next_p]

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
        old_point_index = get_index(old_point,curr_seq)
        new_point = df.loc[i][1][1]
        new_point_index = get_index(new_point, curr_seq)
        curr_seq = modify_sequence(curr_seq, old_point_index, new_point_index)[0]

    return curr_seq


def K_OPT(original_seq, dis_matrix, start_origin_index, distance):

    df = pd.DataFrame(columns=['fun_val', 'switch points'])

    curr_seq = original_seq.copy()

    #  max number of iteration the find the k_opt
    num_iter = 20
    iter_counter = 0

    # the index of the point in curr_seq with the longest edge
    origin_index = start_origin_index

    # the next point to the origin
    if origin_index < len(curr_seq) - 1:
        next_origin_old_index = origin_index + 1
    else:
        next_origin_old_index = 0

    edge_dis = distance
    while True:
        # finding shorter edge from the point 'next_origin_old_index'
        # and return the point that points to to the closes point to 'next_origin_old_index'
        next_origin_new_index = get_shorter_edge(dis_matrix, curr_seq, origin_index, next_origin_old_index, edge_dis)

        if iter_counter >= num_iter or next_origin_new_index is None:
            break

        new_seq, new_origin_index, next_point_index = modify_sequence(curr_seq, next_origin_old_index, next_origin_new_index)

        fun_value = calc_fun_value(new_seq, dis_matrix)

        df = df.append({'fun_val': fun_value, 'switch points': [curr_seq[next_origin_old_index], curr_seq[next_origin_new_index]]}, ignore_index=True)

        # update the variable to the next loop
        curr_seq = new_seq
        origin_index = new_origin_index
        next_origin_old_index = next_point_index
        edge_dis = dis_matrix[curr_seq[origin_index]][curr_seq[next_origin_old_index]]
        iter_counter += 1

    print("num if iteration = ", iter_counter)
    # modify the seq according the k opt, k is the index of the row in the df
    if len(df) > 0:
        k_index = find_min_value(df)
        # create the current optimal seq
        opt_seq = create_opt_seq(df, original_seq, start_origin_index, k_index)
        return opt_seq
    else:
        return None


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # create initial by connecting the point according to the order of points list
    #initial_sequence = initial_solution(points)
    initial_sequence = [6,7,8,9,10,0,1,2,3,4,5]
    dis_matrix = create_distance_matrix(points)
    index = 1
    edge_dis = dis_matrix[7][8]

    seq = K_OPT(initial_sequence, dis_matrix, index, edge_dis)

    solution = seq






    # build a trivial solution
    # visit the nodes in the order they appear in the file
    #solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))



    return output_data









if __name__ == '__main__':
    import sys

    file_location = r"data/tsp_20_1"
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))
