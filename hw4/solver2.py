#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
from pulp import *


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


def define_cons1(model, x, y, facilities, customers):

    for c in range(len(customers)):
        for f in range(len(facilities)):
            model += y[(f, c)] <= x[f]


def define_cons2(model, y, facilities, customers):

    for c in range(len(customers)):
        model += lpSum(y[(f, c)] for f in range(len(facilities))) == 1


def define_cons3(model, y, facilities, customers):

    for f in range(len(facilities)):
        curr_capacity = facilities[f].capacity
        model += lpSum(y[(f, c)] for c in range(len(customers))) <= curr_capacity


def output_customer(y, facilities, customers):

    output = []

    for c in range(len(customers)):
        for f in range(len(customers)):
            if y[(f, c)].varValue == 1:
                output.append(f)
                break
    return output


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

    # define the distance matrix between costumer of facility
    # the row are the facility and the columns are the customers
    dis_matrix = create_dis_matrix(facilities, customers)

    model = LpProblem("Minimize Staffing", LpMinimize)

    # Define Decision Variables
    x = LpVariable.dicts("facility", list(range(len(facilities))), cat='Binary')

    y = LpVariable.dicts("customer", [(f, c) for f in range(len(facilities))
                                      for c in range(len(customers))],cat='Binary')

    # Define Constraint
    define_cons1(model, x, y, facilities, customers)  #מחסן יכול לשרת לקוח רק אם הוא פתוח
    define_cons2(model, y, facilities, customers)  # לקוח יכול לקבל משלוח ממחסן אחד
    define_cons3(model, y, facilities, customers)  # תפוקת המחסן עומדת בסה"כ דרישות הלקוחות

    # Define Objective
    model += (lpSum([facilities[f].setup_cost * x[f] for f in range(len(facilities))])
              + lpSum([dis_matrix[f][c] * y[(f, c)] for f in range(len(facilities)) for c in range(len(customers))]))

    model.solve()
    print("facility variable")
    for f in range(len(facilities)):
        print("{} status {}".format(f, x[f].varValue))

    print("customer variable")
    for f in range(len(facilities)):
        for c in range(len(customers)):
            print("customer {} {} status {}".format(f, c, y[(f, c)].varValue))

    print(value(model.objective))

    output_customer_list = output_customer(y, facilities, customers)

    print(output_customer_list)

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
    output_data = '%.2f' % value(model.objective) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, output_customer_list))

    return output_data


import sys

if __name__ == '__main__':
    import sys

    file_location = r"data/fl_4_1"
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))
