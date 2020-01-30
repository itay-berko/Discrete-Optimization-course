#!/usr/bin/python
# -*- coding: utf-8 -*-

#########
# sort the item according the value density (value/weight)
########

from collections import namedtuple
from operator import attrgetter
import numpy

Item = namedtuple("Item", ['index', 'value', 'weight', 'density'])


# creating the dynamic table
def create_dynamic_table(item_count, capacity, items):

    table = numpy.zeros((capacity + 1, item_count + 1))

    # Initial the first column of the table
    for i in range(len(table)):
        table[i][0] = 0

    # adding values to the table
    for j in range(1, len(table[0])):
        for i in range(len(table)):

            if items[j - 1].weight > i:
                # the current cell will get the value of the last column
                table[i][j] = table[i][j-1]
                continue

            # now i need to decide if i will take the current item or not!
            # if i will chose to take the current item,
            # i need to add the values from the last column in the table to the current one
            # the index of the previous val = current capacity - current weight
            index_curr_capacity_minus_curr_weight = i - items[j - 1].weight

            # if i will chose this item, the tot value will be 'curr_tot_val'
            curr_tot_val = table[index_curr_capacity_minus_curr_weight][j-1] + items[j - 1].value

            # now i need to compare the curr_tot_val to the last value with the same capacity.
            # I will take the max value!

            if table[i][j-1] > curr_tot_val:
                table[i][j] = table[i][j-1]
            else:
                table[i][j] = curr_tot_val

    return table


# return which elements we selected:
def res_dynamic_table(table, items):

    # the index of the last column in the table
    col_index = len(table[0]) - 1
    # Initial the row index
    row_index = len(table) - 1
    selected_items_list = [0]*len(items)

    while col_index > 0:

        current = table[row_index][col_index]
        last = table[row_index][col_index - 1]

        if current != last:
            # I chose this item!
            # need to update the row index location
            row_index = row_index - items[col_index - 1].weight
            selected_items_list[col_index - 1] = 1

        col_index += -1

    value = int(table[len(table) - 1][len(table[0]) - 1])
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, selected_items_list))

    return output_data







def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1]), int(parts[0])/int(parts[1])))


    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0] * len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    table = create_dynamic_table(item_count, capacity, items)
    result = res_dynamic_table(table, items)

    return result


if __name__ == '__main__':
    file_location = r"data/ks_10000_0"

    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()

    print(input_data)

    print(solve_it(input_data))

