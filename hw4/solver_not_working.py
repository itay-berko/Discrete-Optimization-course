#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
from scipy.optimize import linprog


def create_ub(facilities, customers):
    facility_count = len(facilities)
    customer_count = len(customers)
    num_columns = facility_count + facility_count * customer_count
    num_rows = facility_count + facility_count * customer_count
    # the matrix of all the upper bond condition
    # the first columns are the Ywc
    # the columns ofter are the Xw
    A_ub = np.zeros([num_rows, num_columns])
    B_ub = np.zeros([num_rows])

    # condition 1: Yw,c < Xw --> we can't assign costumer to facility ,
    # if the facility is not open!
    # for each row, only one Yw,c = 1 and the others 0
    y_temp = np.identity(facility_count * customer_count)

    # creating multi negative  unit matrix as the number of facilities
    x_temp = np.identity(facility_count) * -1
    x_temp2 = x_temp
    for i in range(1, customer_count):
        x_temp2 = np.vstack((x_temp2, x_temp))

    # horizontal stacking the matrices
    A_ub_temp = np.hstack((y_temp, x_temp2))
    # add the A_ub_temp to the A_ub matrix
    for row in range(len(A_ub_temp[:, 0])):
        for col in range(len(A_ub_temp[0, :])):
            A_ub[row][col] = A_ub_temp[row][col]

        B_ub[row] = 0

    # the index of the row to start enter condition 2
    last_row = len(A_ub_temp[:, 0])

    # condition 2 : the capacity limitation, the rows for that condition
    # will be as the number of the facilities
    facility_index = 0

    for row in range(last_row, len(A_ub[:, 0])):
        col_index = 0
        customer_index = 0
        for c in range(customer_count):
            curr_demand = customers[customer_index].demand
            for f in range(facility_count):
                A_ub[row, col_index] = curr_demand
                col_index += 1
            customer_index += 1

        B_ub[row] = facilities[facility_index].capacity
        facility_index += 1

    return A_ub, B_ub


def create_eq(facilities, customers):
    facility_count = len(facilities)
    customer_count = len(customers)
    num_columns = facility_count + facility_count * customer_count
    num_rows = customer_count

    a_eq = np.zeros([num_rows, num_columns])
    b_eq = np.zeros([num_rows])
    col_index = 0
    for row in range(num_rows):
        for f in range(facility_count):
            a_eq[row, col_index] = 1
            col_index += 1

        b_eq[row] = 1

    return a_eq, b_eq


# the distance matrix between customer to facility
def create_dis_matrix(facilities, customers):
    facility_count = len(facilities)
    customer_count = len(customers)

    dis_matrix = np.zeros([facility_count, customer_count])

    for f in range(facility_count):
        f_point = facilities[f].location
        for c in range(customer_count):
            c_point = customers[c].location
            dis_matrix[f][c] = length(f_point, c_point)

    return dis_matrix


def objective_function(facilities, customers):
    facility_count = len(facilities)
    customer_count = len(customers)
    num_columns = facility_count + facility_count * customer_count
    obj_f = np.zeros([num_columns])

    # the distance matrix between customer to facility
    # rows --> facility
    # column --> customer
    dis_matrix = create_dis_matrix(facilities, customers)

    # add all coefficients to Ywc --> f(w,c), the distance between to customer to facility
    index = 0
    for c in range(customer_count):
        for f in range(facility_count):
            curr_dis = dis_matrix[f, c]
            obj_f[index] = curr_dis
            index += 1

    # add all the coefficients to Xw --> Sf the cost of the facility
    f_index = 0
    for c in range(index, num_columns):
        obj_f[c] = facilities[f_index].setup_cost

    return obj_f


def define_bounds(facilities, customers):
    facility_count = len(facilities)
    customer_count = len(customers)
    num_columns = facility_count + facility_count * customer_count
    bounds = []

    for i in range(num_columns):
        bounds.append((0, 1))

    return bounds


Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    a_ub, b_ub = create_ub(facilities, customers)
    a_eq, b_eq = create_eq(facilities, customers)
    obj_f = objective_function(facilities, customers)
    bounds = define_bounds(facilities, customers)

    res = linprog(obj_f, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method='interior-point')

    print(res)

    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = [-1] * len(customers)
    capacity_remaining = [f.capacity for f in facilities]

    facility_index = 0
    for customer in customers:
        if capacity_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand

    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    # calculate the cost of the solution
    obj = sum([f.setup_cost * used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys

    file_location = r"data/fl_4_1"
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))
