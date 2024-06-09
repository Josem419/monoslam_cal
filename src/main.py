#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu


from math import dist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qrcode import make
from sympy import print_rcode
import cv2
import csv
import yaml
from os import path
import pathlib

# local imports
from camera import *
from data_manager import *
from slam_system import *  # needed to specify the whoel namespace because of name conflicts
from pose import *

from typing import Optional


def makeCamera(
    calib: yaml.YAMLObject, pose: Pose, name: Optional[str] = None
) -> PinholeCamera:
    """
    Return a pinhole camera given a YAML object container a calibraition and a name
    """

    intrinsics = np.array(
        [
            [calib["intrinsics"][0], 0, calib["intrinsics"][2]],
            [0, calib["intrinsics"][1], calib["intrinsics"][3]],
            [0, 0, 1],
        ]
    )

    dist_vec = np.array(calib["distortion_coefficients"])
    # make it a full 5 elements in the vector
    dist_vec = np.append(dist_vec, 0)

    res = np.array(calib["resolution"])
    width, height = res[0], res[1]

    if name is not None:
        frame_id = name
    else:
        frame_id = "cam_0"

    return PinholeCamera(frame_id, width, height, intrinsics, dist_vec, pose)


def run():
    """
    Run the main SLAM program on the data sets defined below
    """

    # setup the data paths
    # data paths
    camera_0_csv = "/home/jy351e/Stanford/CS231A/project/data/MH01/mav0/cam0/data.csv"
    camera_0_calib_file = (
        "/home/jy351e/Stanford/CS231A/project/data/MH01/mav0/cam0/sensor.yaml"
    )

    # TODO use this data unstead of the hardocded IMU settings
    imu_csv = "/home/jy351e/Stanford/CS231A/project/data/MH01/mav0/imu0/data.csv"

    # load camera calibration
    with open(camera_0_calib_file, "r") as f:
        calib = yaml.full_load(f)

    # grab the extrinsics - the file above provides a 4x4 homogenous matrix  (only 3x4 shouold be necessary)
    # comes out as a vector, make sure to reshape to 4x4
    # these are the extrinsics relative to the IMU, so the transform is pretty constnt
    imu2cam_RT = np.array(calib["T_BS"]["data"]).reshape((4, 4))
    # make a pose objte out of this
    cam0_pose = Pose(imu2cam_RT[:3, 3], imu2cam_RT[:3, :3])

    # make a camera object
    cam0: PinholeCamera = makeCamera(calib, cam0_pose)

    # create the some constant classes like the SLAM system or the camera that will be used
    slam_config = SLAMConfig()
    slam_config.nrcams = 1

    slam_config.cameras.append(cam0)
    slam_config.camera_fps = calib["rate_hz"]

    slam_sys = SLAMSystem(slam_config)

    # start loop through the csv's
    # pandas is more flexible than the base csv import
    cam_data_frame = pd.read_csv(camera_0_csv, header=0)
    imu_data_frame = pd.read_csv(imu_csv, header=0)

    # to not introduce any latency pre-parse the imu data intoa synchronized row with the image
    # merge the two on the timestamp key 
    combined_input_data = pd.merge(imu_data_frame, cam_data_frame)

    for index, row in combined_input_data.iterrows():
        # convert nano seconds to seconds
        timestamp = row[0] * 1e-9

        # read in the image
        data_path = path.dirname(camera_0_calib_file) + "/data"
        img = cv2.imread(data_path + "/" + row["filename"])
        imgs = []
        imgs.append(img)

        # read the IMU values for passing into the slam system
        accumulated_imu = []
        measurement = np.array([row[1],row[2],row[3],row[4],row[5],row[6]])
        # covariance values are pulled straight from the config file for the imu
        # TODO: use a YAML object for settign instead of hardcoding
        covariance = np.diag([1.6968e-04, 1.6968e-04, 1.6968e-04,2.0000e-3,2.0000e-3,2.0000e-3])


        accumulated_imu.append([timestamp,measurement,covariance])

        output: SLAMOutput = slam_sys.process(timestamp, imgs, accumulated_imu, accumulated_imu[0][1])

        # # the orb detector is gonna get moved to the tracking/mapping class
        # # initialize an orb feature detector
        # orb = cv2.ORB_create(
        #     nfeatures=2500,
        #     scaleFactor=1.2,
        #     nlevels=8,
        #     edgeThreshold=31,
        #     firstLevel=0,
        #     WTA_K=2,
        #     scoreType=cv2.ORB_HARRIS_SCORE,
        #     patchSize=31,
        #     fastThreshold=20,
        # )

        # # compute descriptors and draw them
        # keypoints, descriptor = orb.detectAndCompute(img, None)

        # visualize and save output for this frame
        if output is None:
            continue


if __name__ == "__main__":

    print("Starting SLAM System")

    run()
