"""
the knapsack

maximum weight 20,000 lbs

product     weight        profitability
A           12800         77878
B           10900         82713
C           11400         82728
D           2100          68423
E           11300         84119
F           2300          77765

the problem:
select most profitable product to ship without exceeding weight limit

decision variables:
Xi = 1 if product i is selected else 0

objective:
maximize z = sum(profitability Yi*Xi)

constraint:
sum(weight,i * Xi) < 20,000

"""

from pulp import *


prod = ['A', 'B', 'C', 'D', 'E', 'F']
weight = {'A': 12800, 'B': 10900, 'C': 11400, 'D': 2100, 'E': 11300, 'F': 2300}
prof = {'A': 77878, 'B': 82713, 'C': 82728, 'D': 68423, 'E': 84119, 'F': 77765}

# Initialize Class
model = LpProblem("Loading Truck Problem", LpMaximize)

# Define Decision Variables
x = LpVariable.dicts('ship_', prod, cat='Binary')

# Define Objective
model += lpSum([prof[i]*x[i] for i in prod])

# Define Constraint
model += lpSum([weight[i]*x[i] for i in prod]) <= 20000
model += x['E'] + x['D'] <= 1  # either product E is selected ot product D is selected,but no both
model += x['D'] <= x['B']   # if product D is selected then product B must also be selected


"""
Other logical constraints

LogicalConstraint                                                      Constraint
-----------------                                                      ----------
If item i is selected,then item j is also selected.                    Xi - Xj ≤ 0
Either item i is selected or item j is selected, but not both.         Xi + Xj = 1
If item i is selected,then item j is not selected.                     Xi - Xj ≤ 1
If item i is not selected,then item j is not selected.                -Xi + Xj ≤ 0
At most one of items i, j, and k are selected.                         Xi + Xj + Xk ≤ 1

"""



# Solve Model
model.solve()
for i in prod:
    print("{} status {}".format(i, x[i].varValue))
