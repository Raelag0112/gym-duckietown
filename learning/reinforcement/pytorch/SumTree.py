import numpy as np
import random
        
class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity

        # Round capacity to 2^n (could be done using lb instead)
        self.tree_depth = 1
        self.actual_capacity = 1
        while self.actual_capacity < capacity:
            self.actual_capacity *= 2
            self.tree_depth += 1

        self.tree_nodes = [np.zeros(2 ** i) for i in range(self.tree_depth)]
        self.start_index = -1

    def append(self, p):
        self.start_index = (self.start_index + 1) % self.capacity
        self.set(self.start_index, p)

    def get(self, i):
        return self.tree_nodes[-1][i]

    def set(self, i, p):
        self.tree_nodes[-1][i] = p

        # Update sums
        for j in range(self.tree_depth - 2, -1, -1):
            i //= 2
            self.tree_nodes[j][i] = (self.tree_nodes[j + 1][2 * i] +
                                     self.tree_nodes[j + 1][2 * i + 1])

    def set_multiple(self, indices, ps):
        # TODO: Smarter update which sets all and recalculates range as needed
        for i, p in zip(indices, ps):
            self.set(i, p)

    def total_sum(self):
        return self.tree_nodes[0][0]

    def index(self, p):
        i = 0
        for j in range(self.tree_depth - 1):
            left = self.tree_nodes[j + 1][2 * i]
            if p < left:
                i = 2 * i
            else:
                p = p - left
                i = 2 * i + 1
        return i

    def sample(self, size):
        indices = []
        bins = np.linspace(0, self.total_sum(), size + 1)
        for a, b in zip(bins, bins[1:]):
            # There's a chance we'll sample the same index more than once
            indices.append(self.index(np.random.uniform(a, b)))

        return indices
