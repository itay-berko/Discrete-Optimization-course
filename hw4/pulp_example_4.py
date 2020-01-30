"""
scheduling problem

expected demand:

day               drivers needed
0 = monday        11
1 = tuesday       14
2 = wednesday     23
3 = thursday      21
4 = friday        20
5 = saturday      15
6 = sunday        8

question:
How many drivers, in total, do we need to
hire?

constraint:
each driver works for 5 consecutive days, followed by 2 days off,
repeated weekly

step               definition
decision var       Xi = number of drivers working on day i
objective          minimize z = X0 + X1 + X2 + X3 + X4 + X5 + X6
subject to         X0 + X3 + X4 + X5 + X6 ≥ 11
                   X0 + X1 + X4 + X5 + X6 ≥ 14
                   *
                   *
                   *
                   X2 + X3 + X4 + X5 + X6 ≥ 8

"""

from pulp import *
import pulp as p
import numpy as np
import pandas as pd

model = p.LpProblem("Minimize Staffing", LpMinimize)

# Define Decision Variables
days = list(range(7))
x = p.LpVariable.dicts("staff_", days, lowBound=0, cat="Integer")

# Define Objective
model += p.lpSum([x[i] for i in days])

# Define Constraints
model += x[0] + x[3] + x[4] + x[5] + x[6] >= 11
model += x[0] + x[1] + x[4] + x[5] + x[6] >= 14
model += x[0] + x[1] + x[2] + x[5] + x[6] >= 23
model += x[0] + x[1] + x[2] + x[3] + x[6] >= 21
model += x[0] + x[1] + x[2] + x[3] + x[4] >= 20
model += x[1] + x[2] + x[3] + x[4] + x[5] >= 15
model += x[2] + x[3] + x[4] + x[5] + x[6] >= 8

# Solve Model
status = model.solve()
print(p.LpStatus[status])

for i in days:
    print(f"Day {i} number of drives: {x[i].varValue}")






