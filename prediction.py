import os
import rrcf
import pandas as pd

class rrcf_stream: 
    def __init__(self, ds_name, num_trees=100, shingle_size=18, tree_size=256): 
        self.ds_name = ds_name

        # Set tree parameters
        self.num_trees = num_trees
        self.shingle_size = shingle_size
        self.tree_size = tree_size
        # Create a forest of empty trees
        self.forest = []
        for _ in range(self.num_trees):
            tree = rrcf.RCTree()
            self.forest.append(tree)
        # Initialize variables for shingling
        self.shingle_buffer = []
        self.index = 0

    #def score_with_rrcf(self, index, value):
    def score_with_rrcf(self, timeIndex, sample):  
        # Create a dict to store anomaly score of each point
        avg_codisp = 0

        # for each tree in the forest...
        self.shingle_buffer.append(sample)
        if len(self.shingle_buffer) < self.shingle_size:
            return None
        self.index += 1
        current_shingle = tuple(self.shingle_buffer[-self.shingle_size:])
        for tree in self.forest:
            # if tree is above permitted size, drop the oldest point (FIFO)
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self.index - self.tree_size)
            tree.insert_point(current_shingle, index=self.index)
            # compute codisp on the new point and take the average among all trees
            avg_codisp += tree.codisp(self.index) / self.num_trees 
        return timeIndex, sample, avg_codisp
    

    