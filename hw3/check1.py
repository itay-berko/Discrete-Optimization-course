def second():
    global x
    x = 10

def first():
    global x
    global x = 2
    second()
    x = 15
    return x


print(first())
