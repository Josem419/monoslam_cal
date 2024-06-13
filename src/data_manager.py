#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu

from dataclasses import dataclass, field
import time
from threading import Lock
from queue import Queue
import gtsam
import numpy as np
from sympy import true
from camera import Camera
from visualizer import *
from typing import Optional, List, Dict


class SLAMConfig:

    def __init__(self) -> None:
        self.nrcams: Optional[int] = None
        self.cameras: List[Camera] = []
        self.index: int = 0
        self.camera_fps: int = 0

        self.imu_params: Optional[gtsam.PreintegrationParams] = None
        self.gt_kps: Optional[np.ndarray] = None
        self.reset: bool = False


class SLAMData:

    def __init__(self) -> None:

        # This whole class serves as a container for data that needs to
        # be passed back and forth between other classes for processign

        self.process_queue = Queue()
        self.local_mapping_proc: bool = False
        self.MKFs = []  # MultiKeyFrames
        self.points = []  # list of map points
        self.currentMap = None
        self.accumulated_imus = []

class SLAMOutput:

    def __init__(self, imgs) -> None:
        self.imgs = imgs
        self.pts = []
        self.pose = None

        self.extra = None


class singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(singleton, cls).__call__(*args, **kwargs)
        return cls.instance


class DataManager(metaclass=singleton):
    def __init__(self):
        self.shutdown = False
        self.data = None
        self.config: SLAMConfig = SLAMConfig()
        self.output = None
        self.mutex = Lock()

        self.visualizer = Visualizer()

        self.print_buffer = ""
        self.first_time_id = None
        self.times = {}
        self.durations = {}
        self.data = SLAMData()

    def set_loop_data(self, imgs):
        """
        Resets output data (profiling, printing, outputs)
        """
        self.print_buffer = ""
        self.first_time_id = None
        self.times = {}
        self.durations = {}

        self.output = SLAMOutput(imgs)

    def reset_data(self):
        self.data = SLAMData()

    def update_data(self, **kwargs):
        self.mutex.acquire()
        for key, value in kwargs.items():
            if key in self.data.__dict__.keys():
                self.data.__dict__[key] = value
        self.mutex.release()

    def update_config(self, **kwargs):
        self.mutex.acquire()
        for key, value in kwargs.items():
            if key in self.config.__dict__.keys():
                self.config.__dict__[key] = value
            else:
                print("[SLAM] Data Manager: invalid config key given")
        self.mutex.release()

    def draw_features(self,cam,features,color, size=11, thickness=1):

        # TODO in a separate thread

        self.output.imgs[cam] = self.visualizer.draw_features(self.output.imgs[cam], features, color, size, thickness)

        if True:
            cv2.imshow("image:", self.output.imgs[cam])
            cv2.waitKey()
