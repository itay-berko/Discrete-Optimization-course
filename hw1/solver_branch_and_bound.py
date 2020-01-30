from collections import namedtuple
from operator import attrgetter
import numpy

Item = namedtuple("Item", ['index', 'value', 'weight', 'density'])


class tree_node:
    def __init__(self, parent, value, room, estimate, take):

        self.value = value
        self.room = room
        self.estimate = estimate
        self.parent = parent
        self.take = take
        self.left = None
        self.right = None

    def set_left_child(self, item):
        self.left = item

    def set_right_child(self, item):
        self.right = item


def relax_estimate(items, capacity):

    curr_capacity = 0
    index = 0
    estimate = 0

    while True:
        if curr_capacity + items[index].weight > capacity or index >= len(items):
            break
        else:

            estimate += items[index].value
            curr_capacity += items[index].weight
            index += 1

    if curr_capacity < capacity:
        ratio = (capacity - curr_capacity)/items[index].weight
        estimate += ratio * items[index].value

    return estimate


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
        items.append(Item(i - 1, int(parts[0]), int(parts[1]), int(parts[0]) / int(parts[1])))

    items2 = sorted(items, key=attrgetter('density'), reverse=True)

    # optimistic estimate
    #estimate = 0
    #for i in items:
        #estimate += i.value


    # relax estimate
    estimate = relax_estimate(items2, capacity)

    print(estimate)

    root = tree_node(None, 0,capacity, estimate, True)

    opt_node = None
    max_root_estimate = 0
    max_depth = 0

    def create_tree(depth, curr_node, value, room, estimate, take):

        nonlocal max_root_estimate

        nonlocal opt_node

        nonlocal max_depth

        nonlocal items

        nonlocal  capacity

        if depth == len(items):
            print("enter the max depth")
            if opt_node is None:
                opt_node = curr_node
                max_depth = depth
            elif value > opt_node.value:
                opt_node = curr_node
                max_depth = depth
            if max_root_estimate < estimate:
                max_root_estimate = estimate
            return



        # taking the next item, the left side of the current node
        room_left = room - items[depth].weight
        # only if there is enough room!!
        if room_left >= 0:
            left_take = True
            value_left = items[depth].value + value
            estimate_left = estimate
            left_node = tree_node(curr_node, value_left, room_left, estimate_left, left_take)
            curr_node.set_left_child(left_node)
            # expand the tree throw the left node
            # create_tree(items, depth, curr_node, value, room, estimate, take)
            create_tree(depth + 1, left_node, value_left, room_left, estimate_left, left_take)

        # not taking the next item, the right side to the current node
        # I will create the right node only if the estimate is larger the max_root_estimate
        estimate_right = estimate - items[depth].value
        if estimate_right > max_root_estimate or items[depth].weight > capacity:
            right_take = False
            value_right = value
            room_right = room
            right_node = tree_node(curr_node, value_right, room_right, estimate_right, right_take)
            curr_node.set_right_child(right_node)
            # expand the tree throw the right node
            create_tree(depth + 1, right_node, value_right, room_right, estimate_right, right_take)
        else:
            print("enter estimate condition in depth = ", depth)
            if opt_node is None:
                opt_node = curr_node
                max_depth = depth
            elif value > opt_node.value:
                opt_node = curr_node
                max_depth = depth
            if max_root_estimate < estimate:
                max_root_estimate = estimate
            return

        return

    create_tree(0, root, 0, capacity, estimate, True)

    taken = []

    def tree_result(curr_node):

        nonlocal taken

        if curr_node.parent is None:
            return

        if curr_node.take:
            taken.insert(0, 1)
        else:
            taken.insert(0, 0)

        tree_result(curr_node.parent)

    tree_result(opt_node)

    def add_the_missing_values():
        nonlocal max_depth
        nonlocal items
        nonlocal taken
        if max_depth < len(items):
            last_index = len(taken) - 1
            for i in range(last_index, len(items) - 1):
                taken.append(0)

    add_the_missing_values()
    #value = 0
    value = opt_node.value

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    file_location = r"data/ks_1000_0"

    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()

    #print(input_data)

    print(solve_it(input_data))