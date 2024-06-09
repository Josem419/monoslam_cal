#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu


import yaml
import gtsam
from threading import Thread
import multiprocessing
import math

from data_manager import DataManager, SLAMConfig
from pose import Pose
from camera import *
from tracker import Tracker
from mapping import LocalMapper
from multiframe import MultiFrame

from itertools import compress

from typing import List, Optional, Union


class SLAMSystem:

    def __init__(self, config: SLAMConfig) -> None:
        """
        config: cameras, feature detector (ORB) & vocab,

        parse config
        load config into the data manager so its available to all
        that request and instance i.e
            tracker
            mapper
            loopcloser

        start threads:
            tracking fast update threa
            mapping thread(s)

        """

        # grab or create the singleton
        self.data_manager = DataManager()
        self.load_config_from_object(config)

        self.tracker = Tracker()
        self.localMapper = LocalMapper()

        # self.t_mapping = Thread(target=self.localMapper.run)
        # self.t_mapping = multiprocessing.Process(target=self.localMapper.run)
        # self.t_mapping.start()

    def stop(self):
        """
        The data manager singleton is the main way to keep the different classes sharing data
        """
        self.data_manager.shutdown = True

    def load_config_from_object(self, config: SLAMConfig) -> None:
        """
        Load the config from a config object. Overwrite existing config
        """

        # set the config in the Data manager
        self.data_manager.update_config(nrcams=config.nrcams)
        self.data_manager.update_config(camera_fps=config.camera_fps)
        self.data_manager.update_config(cameras=config.cameras)

        # TODO make the imu config loading more flexible
        # ============ Setup IMU parameters ============
        imu_params = gtsam.PreintegrationParams.MakeSharedU(
            0
        )  # gravity along negative z axis
        imu_params.setAccelerometerCovariance(
            0.001 * np.eye(3)
        )  # acc white noise in continuous
        imu_params.setIntegrationCovariance(
            1e-7 * np.eye(3)
        )  # integration uncertainty continuous
        imu_params.setGyroscopeCovariance(
            0.0001 * np.eye(3)
        )  # gyro white noise in continuous
        # imu_params.setOmegaCoriolis(w_coriolis)
        self.data_manager.update_config(imu_params=imu_params)

    def load_config_from_file(self, config_filepath) -> None:
        with open(config_filepath, "r") as config_file:
            configs = yaml.safe_load(config_file)

        # ============================== Cameras ==============================
        nrcams = configs["nrcams"]
        cameras = []
        for cam in range(nrcams):
            camera_config = configs["cameras"]["camera{}".format(cam)]
            if camera_config["type"] == "Pinhole":
                T_imu2cam = Pose(
                    np.array(camera_config["extrinsics"]["t"]),
                    np.array(camera_config["extrinsics"]["r"]),
                )
                cameras.append(
                    PinholeCamera(
                        camera_config["frame"],
                        camera_config["intrinsics"]["w"],
                        camera_config["intrinsics"]["h"],
                        np.array(camera_config["intrinsics"]["K"]),
                        np.array(camera_config["intrinsics"]["D"]),
                        T_imu2cam,
                    )
                )

            else:
                print("Other camera types not supported yet")

        self.data_manager.update_config(nrcams=nrcams)
        self.data_manager.update_config(camera_fps=configs["expected_fps"])
        self.data_manager.update_config(cameras=cameras)

        # TODO IMU settings to the config file and object
        # ============ Setup IMU parameters ============
        imu_params = gtsam.PreintegrationParams.MakeSharedU(
            0
        )  # gravity along negative z axis
        
        imu_params.setAccelerometerCovariance(
             2.0000e-3 * np.eye(3)
        )  # acc white noise in continuous
       
        imu_params.setIntegrationCovariance(
            1e-7 * np. eye(3)
        )  # integration uncertainty continuous
        
        imu_params.setGyroscopeCovariance(
            1.6968e-04 * np.eye(3)
        )  # gyro white noise in continuous

        # imu_params.setOmegaCoriolis(w_coriolis)
        self.data_manager.update_config(imu_params=imu_params)

        # ============================== Detectors ==============================

    def process(self, timestamp, imgs, accumulated_imu, interpolated_vals):
        print("SLAMSystem: Starting process Loop")

        if self.data_manager.config.reset:
            print("SLAM system resetting, skipping process loop.")
            return None

        # run a track loop and return all of the
        return self.tracker.track(timestamp, imgs, accumulated_imu, interpolated_vals)

    # TODO
    def updateCalibration(self, calibrations):
        """
        Change the calibration system wide:
        - self.data_manager
        - map and poses in factor graphs

        """
        pass

