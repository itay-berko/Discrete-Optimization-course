from scipy.optimize import linprog

# the problem:
# min -X0 + 4X1
# subject:
# -3X0 + X1 <= 6
# -X0 - 2X1 >= -4
# X1 >= -3

# the vector of the objective function
c = [-1, 4]
# we want to change the constraint in a way: c1*x1 + ....cN*xN <= b
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None,None)
x1_bounds = (-3, None)
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
#print(res)



# the problem:
# min 5X0 + 10X1
# subject:
# 2X0 - X1 >= 10
# X0 + X1 <= 20


c = [5, 10]
A = [[-2, 1], [1, 1]]
b = [-10, 20]
x0_bounds = (0,None)
x1_bounds = (0,None)
res2 = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
print(res2)
