#!/usr/bin/python
# -*- coding: utf-8 -*-


class Node:
    def __init__(self, number):

        self.number = number
        self.neighbors = []
        self.color = None

    def add_neighbor(self, number):
        self.neighbors.append(number)

    def set_color(self,color):
        self.color = color


# create initial list with the same number of nodes
def initial_node_list(node_count):

    node_list = []

    # initial the list with None
    for i in range(node_count):
        node_list.append(None)

    return node_list


# get the node object from the node list according to number (= index in list)
# if the node is None, define the node object
def get_node_object(number, node_list):

    if node_list[number] is None:
        curr = Node(number)
        node_list[number] = curr
        return curr
    else:
        return node_list[number]


# define all the node objects
def define_node(node_count, edges):

    # create None list with the same length as the number of nodes
    node_list = initial_node_list(node_count)

    for i in range(len(edges)):

        # get the node objects at each end of the edge
        node_0 = get_node_object(edges[i][0], node_list)
        node_1 = get_node_object(edges[i][1], node_list)

        # define the neighbors of each node
        node_0.add_neighbor(node_1)
        node_1.add_neighbor(node_0)

    return node_list


# getting the node with the maximum neighbours
def get_node_max_neighbors(node_list):
    max_neighbors = 0
    index_max = 0
    for i in range(len(node_list)):
        if len(node_list[i].neighbors) > max_neighbors:
            max_neighbors = len(node_list[i].neighbors)
            index_max = i

    return node_list[index_max]


# look if the current index is in the list
def in_list(obj, new_list):

    for i in new_list:
        if i == obj:
            return True

    return False


# remove node object from the temp_list
def remove_from_list(obj, curr_list):

    for i in range(len(curr_list)):
        if curr_list[i] == obj:
            del curr_list[i]
            return
    return


# ordering the list according the the number of neighbors
def node_ordering_opt_1(node_list):

    new_list = []

    temp_list = node_list.copy()

    while len(temp_list) > 0:

        curr_obj_max = get_node_max_neighbors(temp_list)
        new_list.append(curr_obj_max)
        remove_from_list(curr_obj_max, temp_list)

    return new_list


def get_node_max_neighbors_in_depth(node_list):

    count_max = 0
    node_max = None

    for node in node_list:

        visited = list()
        visited.append(node)

        neighbors = node.neighbors.copy()
        # the initial depth of the curr_neighbors from the current node
        depth = 2
        # the initial score for this node
        count = len(neighbors)

        next_depth_neighbors = []

        # calc the count value according the number of neighbors * (1/depth)

        while len(neighbors) > 0 or depth < 2:

            if len(neighbors) == 1:

                if not in_list(neighbors[0], visited):
                    # add the neighbors of the current node to the next_depth_list
                    next_depth_neighbors.extend(neighbors[0].neighbors.copy())
                    # add the current node the the visited list
                    visited.append(neighbors[0])
                    count += len(neighbors[0].neighbors) * (1 / depth)
                    del neighbors[0]

                neighbors = next_depth_neighbors.copy()
                next_depth_neighbors = []
                depth += 1

            elif not in_list(neighbors[0], visited):

                # add the neighbors of the current node to the next_depth_list
                next_depth_neighbors.extend(neighbors[0].neighbors.copy())
                # add the current node the the visited list
                visited.append(neighbors[0])
                count += len(neighbors[0].neighbors) * (1 / depth)
                del neighbors[0]

            else:

                del neighbors[0]

        if count > count_max:
            count_max = count
            node_max = node
    print("return node_max")
    return node_max


# ordering the list with respect to the neighbors depth
# depth factor = 1/depth
def node_ordering_opt_2(node_list):

    new_list = []
    temp_list = node_list.copy()

    while len(temp_list) > 0:
        print("in loop")
        curr_obj_max = get_node_max_neighbors_in_depth(temp_list)
        new_list.append(curr_obj_max)
        remove_from_list(curr_obj_max, temp_list)

    return new_list


# define all the option of colors
def create_color_list(node_list):

    color_list = []
    for i in range(len(node_list)):
        color_list.append(i)
    return color_list


def neighbor_with_same_color(color,neighbors):

    for node in neighbors:
        if color == node.color:
            return True

    return False


def assign_colors(node_list):
    # create a list of all the colors options
    color_list = create_color_list(node_list)

    for node in node_list:
        neighbors = node.neighbors
        for color in color_list:
            if neighbor_with_same_color(color, neighbors):
                continue
            else:
                node.set_color(color)
                break


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # create a list of node objects
    print("define nodes")
    node_list = define_node(node_count, edges)
    # sort the list according to the node with the max neighbors.
    #ordering_list_1 = node_ordering_opt_1(node_list)
    print("ordering the list")
    ordering_list_2 = node_ordering_opt_2(node_list)
    print("assign the colors")
    # assign color to each node
    assign_colors(ordering_list_2)

    solution = []

    for node in node_list:
        solution.append(node.color)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys

    file_location = r"data/gc_4_1"
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
        print(solve_it(input_data))


