from pulp import *

problem = LpProblem("problemName", LpMaximize)

# factory cost per day
cf0 = 450
cf1 = 420
cf2 = 400

# factory throughput per day
f0 = 2000
f1 = 1500
f2 = 1000

# production goal
goal = 80000

# time limit
max_num_days = 30

num_factories = 3

# @@define var without bounds
#variable = LpVariable("variableName")

# @@define var with bounds
#var = LpVariable("boundedVariableName", lowerBound, upperBound)

# @@define large number of var of the same type and bounds
#varDict = LpVariable.dicts("varDict", variableNames, lowBound, upBound)

# the decision variables for the computer production problem
# are the days that we spend producing for each factory

num_factories = 3
factory_days = LpVariable.dicts("factoryDays", list(range(num_factories)), 0, 30, cat="Continuous")

# define goal constrain
# 1 - number of units assembled should be above or equal to the goal amount
# 2 - no factory should produce more than double as the other factory
c1 = factory_days[0]*f0 + factory_days[1]*f1 + factory_days[2] * f2 >= goal  # production constraints
c2 = factory_days[0]*f0 <= 2*factory_days[1]*f1
c3 = factory_days[0]*f0 <= 2*factory_days[2]*f2
c4 = factory_days[1]*f1 <= 2*factory_days[2]*f2
c5 = factory_days[1]*f1 <= 2*factory_days[0]*f0
c6 = factory_days[2]*f2 <= 2*factory_days[1]*f1
c7 = factory_days[2]*f2 <= 2*factory_days[0]*f0

# adding the constraints to the problem
problem += c1
problem += c2
problem += c3
problem += c4
problem += c5
problem += c6
problem += c7

# objective function --> minimizing the cost assembling , written as maximizing the negative cost
problem += -factory_days[0]*cf0*f0 - factory_days[1]*cf1*f1 - factory_days[2]*cf2*f2

# solving the problem
problem.solve()

# The solutions to the problem can be obtained by accessing the varValue
# attribute of each variable:
for i in range(3):
 print(f"Factory {i}: {factory_days[i].varValue}")




