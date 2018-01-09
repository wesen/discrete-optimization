#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import functools

from collections import namedtuple

Item = namedtuple("Item", ['index', 'value', 'weight'])


class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            print("Not caching")
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


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
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full

    def algo_naive():
        value = 0
        weight = 0
        taken = [0] * len(items)

        for item in sorted(items, key=lambda x: x.weight):
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight

        return value, weight, taken

    def algo_greedy_value():
        value = 0
        weight = 0
        taken = [0] * len(items)

        for item in sorted(items, key=lambda x: float(x.value), reverse=True):
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight

        return value, weight, taken

    def algo_greedy_value_ratio():
        value = 0
        weight = 0
        taken = [0] * len(items)

        for item in sorted(items, key=lambda x: (float(x.value) / float(x.weight)), reverse=True):
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight

        return value, weight, taken

    def algo_dp_one():
        import sys
        sys.setrecursionlimit(20000)

        @memoized
        def recurse_dp(k, j):
            if j < 0:
                return 0, 0, []
            if k >= items[j].weight:
                v1, w1, t1 = recurse_dp(k - items[j].weight, j - 1)
                v1 += items[j].value
                t1.append(j)

                v2, w2, t2 = recurse_dp(k, j - 1)

                if v1 > v2:
                    return v1, w1, t1
                else:
                    return v2, w2, t2
            else:
                return recurse_dp(k, j - 1)

        return recurse_dp(capacity, len(items) - 1)


    def algo_dp_two():
        import numpy as np
        values = np.zeros((len(items) + 1, capacity+1), dtype=np.uint32)

        for k in range(capacity+1):
            for i in range(len(items)):
                if items[i].weight <= k:
                    v1 = values[i, k]
                    v2 = values[i, k - items[i].weight] + items[i].value
                    v3 = values[i+1, k-1]
                    values[i + 1, k] = max(v1, v2, v3)
                else:
                    values[i + 1, k] = values[i, k]

        taken = [0] * len(items)
        w = 0
        v = 0

        k = capacity
        for i in reversed(range(1, len(items)+1)):
            if values[i, k] != values[i-1, k]:
                v += items[i-1].value
                w += items[i-1].weight
                k -= items[i-1].weight
                taken[i-1] = 1

        return v, w, taken

    max_value = 0

    # value_naive, w, t = algo_naive()
    # value_greedy_value, weight, taken = algo_greedy_value()
    # value_greedy_value_ratio, weight, taken = algo_greedy_value_ratio()

    value_dp_1, weight_dp_1, taken_dp_1 = algo_dp_one()
    taken = [0] * len(items)
    for i in taken_dp_1:
        taken[i] = 1

    max_value = value_dp_1

    # print(value_dp_1, taken)
    #
    # for i in items:
    #     print(i)

    # value_dp_2, weight_dp_2, taken_dp_2 = algo_dp_two()
    # print(value_dp_2, weight_dp_2, taken_dp_2)
    # max_value = value_dp_2
    # taken = taken_dp_2

    # max_value = max([value_naive, value_greedy_value, value_greedy_value_ratio, value_dp_1])

    # if max_value == value_dp_1:
    #     print("dp_1 is best")
    # elif max_value == value_naive:
    #     print("naive is best")
    # elif max_value == value_greedy_value:
    #     print("naive greedy value is best")
    # elif max_value == value_greedy_value_ratio:
    #     print("naive greedy value ratio is best")

    # prepare the solution in the specified output format
    output_data = str(max_value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
