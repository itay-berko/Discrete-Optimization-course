#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import copy
import numpy
import pandas as pd
import numpy as np
import random
import itertools as itr

Point = namedtuple("Point", ['x', 'y'])


def create_random_seq(points,nodeCount):

    seq = []
    for i in range(nodeCount):
        seq.append(i)

    random.shuffle(seq)

    return seq

# calc the tot dis of the seq
def calc_fun_value(seq, dis_matrix, node_count):
    value = 0

    for i in range(0, node_count - 1):
        curr_p = seq[i]
        next_p = seq[i + 1]
        value += dis_matrix[curr_p][next_p]

    value += dis_matrix[next_p][seq[0]]

    return value


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def create_distance_matrix(points):

    dis_matrix = numpy.zeros((len(points),len(points)))

    for i in range(len(points) -1):
        for j in range(i+1,len(points)):

            dis = length(points[i], points[j])
            dis_matrix[i][j] = dis
            dis_matrix[j][i] = dis

    return dis_matrix


def tabu_search(dis_matrix, first_seq, node_count):

    initial_seq = first_seq.copy()

    initial_val = calc_fun_value(initial_seq, dis_matrix, node_count)
    print("the initial solution = ", initial_val)

    Runs = 1000

    ### TABU LIST ###
    Length_of_Tabu_List = 10

    Tabu_List = np.empty((0, node_count + 1))

    One_Final_Guy_Final = []

    Iterations = 1

    Save_Solutions_Here = np.empty((0, node_count + 1))

    for i in range(Runs):
        print()
        print("--> This is the %i" % Iterations, "th Iteration <--")

        # To create all surrounding neighborhood

        List_of_N = list(
            itr.combinations(initial_seq, 2))  # From X0, it shows how many combinations of 2's can it get; 8 choose 2

        Counter_for_N = 0
        # stack  all the seq of the neighbors
        All_N_for_i = []
        # looping all the combination in 'List_of_n' and create seq to each one
        for i in range(len(List_of_N)):
            new_seq = []
            A_Counter = List_of_N[i]  # Goes through each set
            A_1 = A_Counter[0]  # First element
            A_2 = A_Counter[1]  # Second element

            # Making a new list of the new set of departments, with 2-opt (swap)
            for j in range(node_count):  # For elements in X0, swap the set chosen and store it
                if initial_seq[j] == A_1:
                    new_seq.append(A_2)
                elif initial_seq[j] == A_2:
                    new_seq.append((A_1))
                else:
                    new_seq.append(initial_seq[j])

            # stuck all the new seq
            #All_N_for_i = np.vstack((All_N_for_i, new_seq))  # Stack all the combinations
            All_N_for_i.append(new_seq)

        All_N_for_i = np.array(All_N_for_i)

        # stack all the cost value and the seq for each neighbor
        OF_Values_all_N = []
        for curr_seq in All_N_for_i:
            curr_cost = list()
            curr_cost.append(calc_fun_value(curr_seq, dis_matrix, node_count))
            OF_Values_all_N.append(curr_cost + list(curr_seq))

        OF_Values_all_N = np.array(OF_Values_all_N).astype(int)

        # Ordered OF of neighborhood, sorted by OF value
        OF_Values_all_N_Ordered = np.array(sorted(OF_Values_all_N, key=lambda x: x[0]))

        ######################
        ######################

        # Check if solution is already in Tabu list, if yes, choose the next one
        index = 0
        Current_Sol = OF_Values_all_N_Ordered[index]  # Current solution

        while Current_Sol[0] in Tabu_List[:, 0]:  # If current solution is in Tabu list
            index += 1
            Current_Sol = OF_Values_all_N_Ordered[index]

        # If Tabu list is full
        if len(Tabu_List) >= Length_of_Tabu_List:  # If Tabu list is full
            Tabu_List = np.delete(Tabu_List, (Length_of_Tabu_List - 1), axis=0)  # Delete the last row

        Tabu_List = np.vstack((Current_Sol, Tabu_List)).astype(int)

        Save_Solutions_Here = np.vstack(
            (Current_Sol, Save_Solutions_Here)).astype(int)  # Save solutions, which is the best in each run

        # In order to "kick-start" the search when stuck in a local optimum, for diversification

        Mod_Iterations = Iterations % 10

        Ran_1 = np.random.randint(1, node_count + 1)
        Ran_2 = np.random.randint(1, node_count + 1)
        Ran_3 = np.random.randint(1, node_count + 1)

        if Mod_Iterations == 0:
            Xt = []
            A1 = Current_Sol[Ran_1]
            A2 = Current_Sol[Ran_2]

            # the Current_Sol contain [cost, first point seq......last point seq]
            #because of that, i need to add one to the index, to run only on the seq points
            for index2 in range(1, node_count + 1):
                if Current_Sol[index2] == A1:
                    Xt.append(A2)
                elif Current_Sol[index2] == A2:
                    Xt.append(A1)
                else:
                    Xt.append(Current_Sol[index2])


            Current_Sol = Xt

        intitial_seq = Current_Sol[1:]

        Iterations += 1

        # Change length of Tabu List every 5 runs, between 5 and 20, dynamic Tabu list
        if Mod_Iterations == 5 or Mod_Iterations == 0:
            Length_of_Tabu_List = np.random.randint(5, 20)

    t = 0
    Final_Here = []

    min_cost = Save_Solutions_Here[0, 0]

    for index3 in range(len(Save_Solutions_Here)):

        if Save_Solutions_Here[index3, 0] < min_cost:
            Final_Here = Save_Solutions_Here[t, :]




    print()
    print()
    print("DYNAMIC TABU LIST")
    print()
    print("Initial Cost:", initial_val)
    print()
    print("The Lowest Cost is:", Final_Here[0])

    return Final_Here[1:]




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

    dis_matrix = create_distance_matrix(points)
    initial_seq = create_random_seq(points, nodeCount)
    final_seq = tabu_search(dis_matrix, initial_seq, nodeCount)
    solution = final_seq

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
