#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:05:08 2019

@author: jan
"""

import tensorflow as tf
import numpy as np

def gaussian_neighborhood(dist, sigma):
    return tf.exp(-tf.divide(tf.pow(dist, 2.0),
                             tf.multiply(2.0, tf.pow(sigma, 2))))
    