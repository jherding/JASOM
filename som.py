#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:17:43 2019

@author: jan
"""

import tensorflow as tf
import numpy as np
from scipy.spatial import distance_matrix

from distance_metrics import squared_distance
from neighborhood_metrics import gaussian_neighborhood

class SOM:
    """ implement a classic 2-dimensional self-organizing map (SOM) using
    tensorflow
    """
    def __init__(self, m, n, input_dim, dist_func=squared_distance,
                 learning_rate_start=1, learning_rate_end=0.001,
                 nbhood_size_start=None, nbhood_size_stop=0.01,
                 learning_mode='online', max_steps=10000):
        """ constructor
        """
        # define output/map structure (two-dimensional rectangular map)
        self.m = m
        self.n = n
        
        # define input structure
        self.input_dim = input_dim

        # initialize some necessary parameters
        self._learning_mode = learning_mode
        self._sigma_start = np.float32((m*n)/2) if nbhood_size_start is None else np.float32(nbhood_size_start)
        self._sigma_end = np.float32(nbhood_size_stop)
        self._epsilon_start = np.float32(learning_rate_start)
        self._epsilon_end = np.float32(learning_rate_end)
        self._trained = False
        
        self._max_steps = max_steps 
        
        # which distance function to use
        self._dist_func = dist_func

        # initialize everything tensorflow related
        self._graph = tf.Graph()
        self._build_tf_graph()

    def _build_tf_graph(self):
        """ construct a tensorflow graph for the computation to train the SOM
        """
        with self._graph.as_default():
            # initialize the the indices of each neuron in the SOM
            # Note that a single list of indices is not enough since we need
            # to compute the neighbourhood function (in the map/output space)
            # for learning
            loc_list = self._neuron_locations()
            self._map_idx = tf.constant(loc_list,
                                        name='map_idx',
                                        dtype=tf.float32)
            # compute the distance matrix between all neurons in the map
            self._neigh_dist = tf.constant(distance_matrix(loc_list, loc_list, p=1),
                                           name='neighborhood_distances',
                                           dtype=tf.float32)

            # initialize the weights, i.e., the locations of each unit in input space
            self._unit_loc = tf.get_variable(name='unit_loc',
                                             shape=[self.m * self.n, self.input_dim],
                                             initializer=tf.random_normal_initializer())
            
            # initialize the learning rate & neighbourhood size
            self._learning_rate = tf.get_variable(name='learning_rate',
                                                  initializer=tf.constant(self._epsilon_start, dtype=tf.float32))
            self._nbhood_size = tf.get_variable(name='neighborhood_size',
                                                initializer=tf.constant(self._sigma_start, dtype=tf.float32))

            # a placeholder for the input data
            self._in = tf.placeholder(tf.float32,
                                      shape=[1, self.input_dim])
            
            self._t = tf.placeholder(tf.float32,
                                     shape=[])
            
            self._t_max = tf.constant(self._max_steps,
                                      name='max_steps',
                                      dtype=tf.float32)
            
            # create an op to update the SOM
            if self._learning_mode == 'online':
                self._training_op = tf.assign(self._unit_loc, self._online_learning())
            else:
                print('Learning mode not implemented!')
                return
            
    def _online_learning(self):
        """ implement a single update step of SOM learning rule
        """
        input_distances = self._dist_func(self._in, self._unit_loc)
        
        # compute best matching unit for input
        bmu_idx = tf.argmin(input_distances)

        # compute neighbourhood factor
        learning_fac = tf.multiply(
                            gaussian_neighborhood(self._neigh_dist[bmu_idx], self._nbhood_size),
                            self._learning_rate)
        
        # decrease learning rate and neighborhood size (sigma)
        self._learning_rate = tf.multiply(self._epsilon_start,
                                          tf.pow(tf.divide(self._epsilon_end,
                                                           self._epsilon_start),
                                                 tf.divide(self._t, self._t_max)))
        
        self._nbhood_size = tf.multiply(self._sigma_start,
                                        tf.pow(tf.divide(self._sigma_end,
                                                         self._sigma_start),
                                               tf.divide(self._t, self._t_max)))
        
        # compute new locations
        return tf.add(self._unit_loc,
                      tf.multiply(tf.stack([learning_fac, learning_fac], 1),
                                  tf.subtract(self._unit_loc, self._in)))

    def _bmu(self, distances):
        """ find the best matching unit (BMU) for a given sample
        """
        bmu_idx = tf.argmin(distances)
        return tf.gather(self._map_idx, bmu_idx)

    def _neuron_locations(self):
        """ Maps an absolute neuron index to a 2d vector for calculating the
        neighborhood function
        """
        loc_list = np.zeros((self.m*self.n, 2))
        for i in range(self.m):
            for j in range(self.n):
                loc_list[i*self.n + j, :] = np.array([i, j])
        return loc_list        

    def get_graph(self):
        return self._graph

    def get_bmu(self, input_sample):
        """ a public function to get the bmu for a given input
        """        
        return self._bmu(self._dist_func(input_sample, self._unit_loc))
        

    def train(self, data, max_steps=None):
        """ train the SOM
        """
        # make sure that the input data has one sample per row and matchses the
        # expected input dimension
        if data.shape[1] != self.input_dim:
            print('Error! Expected input dimension: {}, data dimension: {}'.format(self.input_dim, data.shape[1]))
            return
        
        with self._graph.as_default():
            if max_steps is not None:
                self._t_max = tf.constant(max_steps,
                                          name='max_steps')

        with tf.Session(graph=self._graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # make sure that enough data is available for max_steps training steps
            # get max_steps samples from the given data in random order
            rand_idx = np.random.choice(np.arange(data.shape[0]),
                                        size=self._t_max.eval())
            training_data = data[rand_idx]
            for t, sample in enumerate(training_data):
                sess.run(self._training_op,
                         feed_dict={self._in: sample.reshape((1,self.input_dim)),
                                    self._t: t})
                if t%100 == 0:
                    print('{:5d} steps done')