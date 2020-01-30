import numpy as np


# method the generate the tubleau matrix
def gen_matrix(var, cons):
    return np.zeros((cons + 1, var + cons + 2))


# check to see if 1 + pivots are required due to a negative element
# in the last column excluding the last row.
def next_round_r(table):
    m = min(table[:-1,-1])
    if m >= 0:
        return False
    else:
        return True


# check if 1+ pivots are required due to a negative element
# in the bottom row (the objective function), excluding the final value
def next_round(table):
    lr = len(table[:,0])
    m = min(table[lr-1,:-1])
    if m >= 0:
        return False
    else:
        return True


# return the row index of the negative value in last column
def find_neg_r(table):
    lc = len(table[0,:])
    m = min(table[:-1,lc-1])
    if m <= 0:
        n = np.where(table[:-1,lc-1] == m)[0][0]
    else:
        n = None
    return n


# return the column index of the negative element in the last row (the objective function)
def find_neg(table):
    lr = len(table[:,0])
    m = min(table[lr-1,:-1])
    if m <= 0:
        n = np.where(table[lr-1,:-1] == m)[0][0]
    else:
        n = None
    return n


# return the pivot element location corresponding to the negative values at the last column
def loc_piv_r(table):
    total = []
    r = find_neg_r(table)
    row = table[r,:-1]
    m = min(row)
    c = np.where(row == m)[0][0]
    col = table[:-1,c]
    for i, b in zip(col,table[:-1,-1]):
        if i**2>0 and b/i>0:
            total.append(b/i)
        else:
            total.append(10000)
    index = total.index(min(total))
    return [index,c]


# return the pivot element location corresponding to the
# negative value at the last row (the objective function)
def loc_piv(table):
    if next_round(table):
        total = []
        n = find_neg(table)
        for i,b in zip(table[:-1,n],table[:-1,-1]):
            if b/i >0 and i**2>0:
                total.append(b/i)
            else:
                total.append(10000)
        index = total.index(min(total))
        return [index,n]


# pivoting method
def pivot(row, col, table):
    lr = len(table[:,0])
    lc = len(table[0,:])
    t = np.zeros((lr,lc))
    pr = table[row,:]
    if table[row,col]**2>0:
        e = 1/table[row,col]
        r = pr*e
        for i in range(len(table[:,col])):
            k = table[i,:]
            c = table[i,col]
            if list(k) == list(pr):
                continue
            else:
                t[i,:] = list(k-r*c)
        t[row,:] = list(r)
        return t
    else:
        print('Cannot pivot on this element.')


# method to enter inequality constrains, for example:
# ('1,3,L,5') means 1*(X1) + 3*(X2) <= 5
# ('1,3,G,5') means 1*(X1) + 3*(X2) >= 5
# @@@@@@@ i didn't understand way we convert to right side also to negative!!
def convert(eq):
    eq = eq.split(',')
    if 'G' in eq:
        g = eq.index('G')
        del eq[g]
        eq = [float(i)*-1 for i in eq]
        return eq
    if 'L' in eq:
        l = eq.index('L')
        del eq[l]
        eq = [float(i) for i in eq]
        return eq


# for min problem convert the objective function
def convert_min(table):
    table[-1,:-2] = [-1*i for i in table[-1,:-2]]
    table[-1,-1] = -1*table[-1,-1]
    return table


# @@@ didn't understand
def gen_var(table):
    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    v = []
    for i in range(var):
        v.append('x'+str(i+1))
    return v


# check if i can add constrains to the matrix
def add_cons(table):
    lr = len(table[:,0])
    empty = []
    for i in range(lr):
        total = 0
        for j in table[i,:]:
            total += j**2
        if total == 0:
            empty.append(total)
    if len(empty)>1:
        return True
    else:
        return False


# at the first example , the size of table is 3X6
def constrain(table,eq):
    if add_cons(table) == True:
        lc = len(table[0,:])
        lr = len(table[:,0])
        var = lc - lr -1
        j = 0
        while j < lr:
            row_check = table[j,:]
            total = 0
            for i in row_check:
                total += float(i**2)
            if total == 0:
                row = row_check
                break
            j +=1
        eq = convert(eq)
        i = 0
        while i<len(eq)-1:
            row[i] = eq[i]
            i +=1
        row[-1] = eq[-1]
        row[var+j] = 1
    else:
        print('Cannot add another constraint.')


def add_obj(table):
    lr = len(table[:,0])
    empty = []
    for i in range(lr):
        total = 0
        for j in table[i,:]:
            total += j**2
        if total == 0:
            empty.append(total)
    if len(empty)==1:
        return True
    else:
        return False


def obj(table,eq):
    if add_obj(table)==True:
        eq = [float(i) for i in eq.split(',')]
        lr = len(table[:,0])
        row = table[lr-1,:]
        i = 0
        while i<len(eq)-1:
            row[i] = eq[i]*-1
            i +=1
        row[-2] = 1
        row[-1] = eq[-1]
    else:
        print('You must finish adding constraints before the objective function can be added.')


def minz(table):
    table = convert_min(table)
    while next_round_r(table)==True:
        table = pivot(loc_piv_r(table)[0],loc_piv_r(table)[1],table)
    while next_round(table)==True:
        table = pivot(loc_piv(table)[0],loc_piv(table)[1],table)
    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    i = 0
    val = {}
    for i in range(var):
        col = table[:,i]
        s = sum(col)
        m = max(col)
        if float(s) == float(m):
            loc = np.where(col == m)[0][0]
            val[gen_var(table)[i]] = table[loc,-1]
        else:
            val[gen_var(table)[i]] = 0
            val['min'] = table[-1,-1]*-1
    return val



if __name__ == "__main__":
    # we have to constraint with two variable
    # variable = x1,x2
    # the method return matrix with number of row according to the number of constrains
    # and number of columns according to the number of variables + slack variable for each constrain
    # + two columns corresponding values (p and b)
    # the size of matrix m will be 3X6
   
    m = gen_matrix(2, 2)
    constrain(m, '2,-1,G,10')
    constrain(m, '1,1,L,20')
    obj(m,'5,10,0')
    print(minz(m))



















