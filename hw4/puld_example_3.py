"""
Capacitated plantlocation model
Modeling
* Production at regional facilities
   - Two plant sizes (low / high)
* Exporting production to other regions
* Production facilities open / close

Decision variables
What we can control:
Xij = quantity produced at location i and shipped to j
Yis = 1 if the plant at location i of capacity s is open, 0 if closed
  # s = low or high capacity plant

Objective function
Minimize z = sum[1 to n](f,is* Yis ) + sum[1 to n]sum[1 to m](c,ij * Xij )
c,ij = cost of producing and shipping from plant i to region j
f,ij = fixed cost of keeping plant i of capacity s open
n= number of production facilities
m = number of markets or regional demand points

"""

from pulp import *
import pulp as p
import numpy as np
import pandas as pd


# Initialize Class
model = p.LpProblem("Capacitated Plant Location Model", LpMinimize)

# Define Decision Variables
loc = ['A', 'B', 'C', 'D', 'E']
size = ['Low_Cap','High_Cap']

x = p.LpVariable.dicts("production", [(i, j) for i in loc for j in loc],
                     lowBound=0, upBound=None, cat='Continous')

y = p.LpVariable.dicts("plant", [(i, s) for s in size for i in loc],
                     cat='Binary')

fix_cost = pd.DataFrame([[1, 2], [3, 4], [1, 2], [6, 7], [5, 2]],
                        index=['A', 'B', 'C', 'D', 'E'],
                        columns=['Low_Cap', 'High_Cap'])


var_cost = pd.DataFrame([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [4, 1, 3, 6, 2],
                        [1, 1, 7, 2, 7], [10, 1, 6, 2, 3]],
                        index=['A', 'B', 'C', 'D', 'E'],
                        columns=['A', 'B', 'C', 'D', 'E'])


# Define objective function
model += (lpSum([fix_cost.loc[i, s]*y[(i, s)] for s in size for i in loc])
          + lpSum([var_cost.loc[i,j]*x[(i,j)] for i in loc for j in loc]))


# Solve Model
status = model.solve()   # Solver
print(p.LpStatus[status])

for i in loc:
    for j in loc:
        print(f"Plant {i} ship to {j}: {x[i,j].varValue}")

for i in loc:
    for j in size:
        print(f"Plant {i} size {j}: {y[i, j].varValue}")
