#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu

import numpy as np


class Feature:
    def __init__(self, keypoint, descriptor) -> None:
        self.keypoint: np.ndarray = keypoint
        self.descriptor: np.ndarray = descriptor
