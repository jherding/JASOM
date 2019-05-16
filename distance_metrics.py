#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:47:07 2019

@author: jan
"""

import tensorflow as tf
import numpy as np

def squared_distance(a, b):
    return tf.reduce_sum(tf.pow(tf.subtract(a, b), 2), 1)

def neuron_locations(m, n):
        """ Maps an absolute neuron index to a 2d vector for calculating the
        neighborhood function
        """
        loc_list = np.zeros((m*n, 2))
        for i in range(m):
            for j in range(n):
                loc_list[i*n + j, :] = np.array([i, j])
        return loc_list
