#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu

# System imports
import numpy as np
from numpy.linalg import inv    # NOTE: export OPENBLAS_NUM_THREADS=1 results in faster compute for small matrices
# from scipy.linalg import inv
import inspect, traceback, os, scipy
import matplotlib.pyplot as plt

# Caravan imports
import rospy



myname = os.path.basename(__file__)

class Kalman():
  def __init__(self) -> None:
    pass

class EKF():
  def __init__(self) -> None:
    pass