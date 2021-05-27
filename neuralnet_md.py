#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##############################################################################
# Multi-layer Perceptron (MLP) implementation
# Matthew Dobson
# github - 
# 
##############################################################################

import numpy as np
import sys

class NeuralNetMLP(object):
    # Feedforward neural network / MLP classifier
    #
    # Parameters
    # ----------
    # n_hidden : int (default: 30)
    #   Number of hidden units.
    # l2 : float (default: 0.)
    #   Lambda value for L2-regularization
    #   No reg if l2 = 0 (default)
