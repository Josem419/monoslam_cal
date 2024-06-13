#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu


from threading import Thread
import time
import sys
import gtsam
import numpy as np
from enum import Enum
import cv2
from pose import Pose
import pymap3d
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions
import math
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching  # scipy>=1.6.0

from itertools import compress
from queue import Queue
import copy
from typing import List, Optional, Union

import matplotlib.pyplot as plt

from data_manager import DataManager
from multiframe import MultiFrame
from mapping import *
from camera import *
from optimizer import Optimizer
from imu import IMU
from utilities import associate_by_triangulation


class State(Enum):
    SYSTEM_NOT_READY = -2
    PAUSED = -1
    NO_IMAGES_YET = 0
    NOT_INITIALIZED = 1
    INITIALIZING = 2
    RUNNING = 3
    LOST = 4

class Tracker:

    def __init__(self) -> None:
        # grab the singleton to share data
        self.data_manager = DataManager()
        self.state = State.NOT_INITIALIZED

        self.currentMultiframe: Optional[MultiFrame] = None
        self.lastMultiframe: Optional[MultiFrame] = None
        self.initMultiframe: Optional[MultiFrame] = None
        self.refMultiKeyframe: Optional[MultiFrame] = None

        self.previous_timestamp: Optional[float] = None

        self.optimizer = Optimizer()
        self.loops = 0
        # Settings to be transferred to data_manager from config file:
        self.settings = {
            "init_min_desc_distance": 300,
            "min_features": 70,
            "init_min_matches": 70,
            "max_reproj_dist": 10000,  # max point triangulation distance
            "init_min_reproj_err": 20,  # init min reprojection error threshold TODO define based on resolution and gt pose accuracy
            "tracking_compare_frame_window": 30,  # tracking: considers mappoints which have been detected < 30 frames ago
            "tracking_compare_image_window": 200,  # square width search region for feature matching
            "tracking_desc_thresh": 630,  # best match min descriptor hamming distance
            "tracking_reproj_thresh": 100,  # best match min reprojection error (based on imu propogated pose) -> should be dependent on that pose covariance...
            "tracking_min_matches": 7,  # min number of features to be matched to keep tracking (tune. Less that this value, the pose estimation should break down)
            "tracking_outlier_reproj_thresh": 7,
        }  # outlier reprojection error threshold

    def track(
        self,
        timestamp: float,
        imgs: List[np.ndarray],
        accumulated_imu: List[List[Union[float, np.ndarray, np.ndarray]]],
        interpolated_vals,
    ):

        print(
            "\n[SLAM] {} - {}".format(self.state.name, self.data_manager.config.index)
        )
        self.data_manager.set_loop_data(imgs)

        if self.state is State.PAUSED:
            return None

        # Process frame - extract the ORB descriptors
        self.currentMultiframe = MultiFrame(
            timestamp, imgs, self.data_manager.config.index
        )

        # C++ sets descriptor dimensions to local mapper

        # Process IMU:
        self.data_manager.data.accumulated_imus.append(
            IMU(
                self.previous_timestamp,
                timestamp,
                accumulated_imu,
                self.data_manager.config.index,
            )
        )
        # ===================================================================

        # TODO mutexing of map

        # Track:
        if self.state is State.NOT_INITIALIZED:

            if self.set_init():
                self.state = State.INITIALIZING

        elif self.state is State.INITIALIZING:

            # TODO multiple frame initialization ...? Not with ground truth at least...
            init_return = self.initialize()
            if init_return == 1:
                self.state = State.RUNNING

                # Add keyframes: TODO use update method in data_manager with mutex
                self.data_manager.data.process_queue.put_nowait(self.initMultiframe)
                self.data_manager.data.process_queue.put_nowait(self.currentMultiframe)
                self.refMultiKeyframe = self.currentMultiframe

            elif init_return == 0:
                #self.reset()
                return None

        elif self.state is State.RUNNING:
            self.data_manager.data.currentMap.mutex.acquire()
            print(" A")
            if self.TrackFrame():
                self.data_manager.data.currentMap.mutex.release()
                print(" R")

                # Determine if MultiKeyFrame needed:
                if self.NeedNewMKF():
                    print("adding KF")
                    self.data_manager.data.process_queue.put_nowait(
                        self.currentMultiframe
                    )
                    self.refMultiKeyframe = self.currentMultiframe
                    print("added KF")

            else:
                self.data_manager.data.currentMap.mutex.release()
                self.reset()
                return None

        print("outputting mappoints")

        # Outputting:
        if self.data_manager.data.currentMap is not None:
            for mappoint in self.data_manager.data.currentMap.mappoints:
                self.data_manager.output.pts.append(mappoint.cartesian)

        self.data_manager.output.pose = self.currentMultiframe.T_origin2body_estimated
        print("outputted mappoints")

        self.previous_timestamp = timestamp
        self.lastMultiframe = self.currentMultiframe
        self.data_manager.config.index += 1

        return self.data_manager.output

    def NeedNewMKF(self):
        """
        If localmapping is not pre-processing
        OR > 1s has passed since last MKF
        OR

        TODO

        also: if only tracking mode, False
        """

        if not self.data_manager.data.local_mapping_proc:
            return True
        return False

    def reset(self):
        """
        Resets Tracker
        """
        print("\n[SLAM] RESETTING ========================================")
        self.state = State.NOT_INITIALIZED
        self.data_manager.config.reset = True

    # =================================== TRACKING METHODS ===================================

    def TrackFrame(self):
        t0 = time.time()

        # Propagate state to current time for prior:
        approx_translation = np.squeeze(
            self.lastMultiframe.T_origin2body_estimated.t.reshape((1, 3))
        ) + self.lastMultiframe.vel_estimated * (
            self.currentMultiframe.timestamp - self.lastMultiframe.timestamp
        )
        self.currentMultiframe.T_origin2body_estimated = Pose(
            approx_translation,
            self.lastMultiframe.T_origin2body_estimated.R,
            np.diag([100, 100, 100, 0.1, 0.1, 0.1]),
        )  # position then rotation
        self.currentMultiframe.vel_estimated = self.lastMultiframe.vel_estimated
        self.currentMultiframe.vel_estimated_covariance = np.diag([5, 5, 5])

        # use ground truth pose:
        # self.currentMultiframe.T_origin2body_estimated = self.currentMultiframe.T_origin2body
        # self.currentMultiframe.T_origin2body_estimated.covariance = np.diag([1,1,1,0.001,0.001,0.001])
        # self.currentMultiframe.vel_estimated = self.currentMultiframe.enu_vel

        data = {}
        data["currentMultiframe"] = self.currentMultiframe
        data["lastMultiframe"] = self.lastMultiframe
        data["accumulated_imus"] = self.data_manager.data.accumulated_imus
        data["map"] = None

        # Optimize the propagated pose with IMU data:
        init_isam, pose_idx, factor_graph, values, fg_ = self.optimizer.PosePredict(
            data
        )
        data["init_isam"] = init_isam
        data["pose_idx"] = pose_idx
        data["factor_graph"] = factor_graph
        data["values"] = values
        data["fg_"] = fg_
        data["map"] = self.data_manager.data.currentMap

        # Draw initialized pose:
        for cam in range(self.data_manager.config.nrcams):
            uv_kpts, valid = self.data_manager.config.cameras[cam].projectPoints(
                self.data_manager.config.gt_kps,
                self.currentMultiframe.T_origin2body_estimated,
            )
            uv_kpts = uv_kpts[valid]
            self.data_manager.draw_features(
                cam, uv_kpts, (255, 255, 0), size=4, thickness=4
            )  # G

        # Feature matching with last n frames:
        n = self.settings[
            "tracking_compare_frame_window"
        ]  # TODO do with timestamp instead
        desc_thresh = self.settings["tracking_desc_thresh"]  # 630
        reproj_thresh = self.settings[
            "tracking_reproj_thresh"
        ]  # 150 # this will depend on the quality of the estimated pose TODO make dependent on estimated pose covariance

        # can log the descritor and pixel distance distributions:
        # desc_loss = []
        # reproj_loss = []

        # filter for mappoints that have been observed in last n frames
        mappoint_indices = np.array(
            [
                i
                for i in range(len(self.data_manager.data.currentMap.mappoints))
                if self.data_manager.data.currentMap.mappoints[i].most_recent_frame
                >= self.currentMultiframe.frame_idx - n
            ]
        )
        if len(mappoint_indices) == 0:
            return 0
        relevant_mapPoints: List[MapPoint] = [
            mapPoint
            for mapPoint in self.data_manager.data.currentMap.mappoints
            if mapPoint.most_recent_frame >= self.currentMultiframe.frame_idx - n
        ]

        # Store association indices for later outlier rejection:
        associated_frame_mappoints_idx = []
        associated_frame_features_idx = []

        # Loop through cameras and associate map features:
        for cam in range(self.data_manager.config.nrcams):
            """
            TODO abstract to function
            Filters for relevant map points that are predicted to be in view using imu propaged pose
            Creates a cost matrix, initializing all terms to high costs
            For each map point:
                filter the features that are within a window TODO see how ORB-SLAM does this, it uses octives and scales...
                fill in the association cost of each of those feature-point associations (based  on descriptor distance and reproj. distance)
            -> solve sparse cost matrix to get assignment matrix
            Add observations to map-points based on if cost is below threshold
            TODO better define cost
            """
            count = 0
            associated_camera_features = []
            associated_camera_mappoints = []
            associated_camera_mappoints_idx = []
            associated_camera_features_idx = []

            camera = self.data_manager.config.cameras[cam]

            # Reproject in this frame using estimated pose
            uv_curr_reproj, valid = camera.projectPoints(
                np.array([mapPoint.cartesian for mapPoint in relevant_mapPoints]),
                self.currentMultiframe.T_origin2body_estimated,
                padding=20,
            )  # use padding for uncertainty in pose
            uv_curr_reproj = uv_curr_reproj[valid]
            if len(uv_curr_reproj) == 0:
                continue  # no visible features, no matching
            inview_relevant_mapPoints = list(compress(relevant_mapPoints, valid))
            inview_mappoint_indices = mappoint_indices[valid]

            cost_matrix = (
                np.ones(
                    (
                        len(self.currentMultiframe.feature_pts[cam]),
                        len(inview_relevant_mapPoints),
                    )
                )
                * 10
            )

            # loop through map points to match to current features:
            for i, mapPoint in enumerate(inview_relevant_mapPoints):

                # search features within window:
                feature_bools = self.currentMultiframe.filter_features(
                    cam,
                    uv_curr_reproj[i],
                    self.settings["tracking_compare_image_window"],
                )

                filtered_features = self.currentMultiframe.feature_pts[cam][
                    feature_bools
                ]
                filtered_desc = self.currentMultiframe.descriptors[cam][feature_bools]
                feature_indices = list(
                    compress(range(len(feature_bools)), feature_bools)
                )

                if len(filtered_features) == 0:
                    continue  # no associations for this mappoint

                # Calculate descriptor and reprojection distance between map point and filtered features:
                mappoint_desc = mapPoint.observations[
                    -1
                ].descriptor  # latest descriptor TODO consider mutlitple frames? and select right descriptor based on camera...
                desc_dist = np.linalg.norm(
                    filtered_desc - mappoint_desc, axis=1
                )  # hamming distance score TODO change to something better
                eucl_dist = np.linalg.norm(
                    filtered_features - uv_curr_reproj[i], axis=1
                )

                # Fill cost matrix column:
                cost_matrix[feature_indices, i] = (
                    eucl_dist / reproj_thresh + desc_dist / desc_thresh
                )  # TODO do a max() instead ? -> Then change below to cost_matrix[assignments_features[i], assignments_points[i]] < 1
                # Faster than:
                # for j, feature_idx in enumerate(feature_indices):
                #     cost_matrix[feature_idx][i] = eucl_dist[j]/reproj_thresh + desc_dist[j]/desc_thresh

            t01 = time.time()
            assignments_features, assignments_points = (
                min_weight_full_bipartite_matching(csr_matrix(cost_matrix))
            )  # this is sparce implementation of scipy.linear_sum_assignment
            # print("assignment: {:.1f} ms".format((time.time()-t01)*1000))

            # Register the assocations by adding observations:
            for i in range(len(assignments_features)):
                if cost_matrix[assignments_features[i], assignments_points[i]] < 2:
                    # feature assignments_features[i] matched to mapPoint assignments_points[i]
                    # assignments_features[i] is of length len(inview_relevant_mapPoints),
                    # but has any values 0-499 (500 features per image)
                    count += 1

                    inview_relevant_mapPoints[assignments_points[i]].addObservation(
                        cam,  # camera id
                        self.currentMultiframe.frame_idx,  # frame id
                        self.currentMultiframe.feature_pts[cam][
                            assignments_features[i]
                        ],  # feature loc
                        self.currentMultiframe.descriptors[cam][
                            assignments_features[i]
                        ],  # feature desc
                        cost_matrix[assignments_features[i], assignments_points[i]],
                    )  # association cost

                    # Keep track of associated map points and features for outlier rejection later:
                    associated_camera_features.append(
                        self.currentMultiframe.feature_pts[cam][assignments_features[i]]
                    )  # just for displaying points and comparing to PnP
                    associated_camera_mappoints.append(
                        inview_relevant_mapPoints[assignments_points[i]].cartesian
                    )
                    associated_camera_mappoints_idx.append(
                        inview_mappoint_indices[assignments_points[i]]
                    )
                    associated_camera_features_idx.append(assignments_features[i])

            associated_frame_mappoints_idx.append(associated_camera_mappoints_idx)
            associated_frame_features_idx.append(associated_camera_features_idx)

            # ============= Displaying points: ================
            # matching map points projected with ground truth to compare
            uv_curr_reproj, valid = camera.projectPoints(
                np.array(
                    [
                        mapPoint.cartesian
                        for mapPoint in self.data_manager.data.currentMap.mappoints
                        if mapPoint.most_recent_frame
                        == self.currentMultiframe.frame_idx
                    ]
                ),
                self.currentMultiframe.T_origin2body,
            )  # NOTE for visualization, ground truth pose is used
            # All mappoints projected with estimated pose:
            # uv_curr_reproj, valid = camera.projectPoints(np.array([mapPoint.cartesian for mapPoint in self.data_manager.data.currentMap.mappoints]),self.currentMultiframe.T_origin2body_estimated)

            uv_curr_reproj = uv_curr_reproj[valid]
            self.data_manager.draw_features(cam, uv_curr_reproj, (0, 255, 0), size=5)

            self.data_manager.draw_features(
                cam, associated_camera_features, (0, 0, 255), thickness=5
            )

            # ================== RANSAC PnP ====================
            # NOTE RANSAC PnP is currently only used to compare with optimized pose below. Could be used for oulier rejection...
            t_pnp0 = time.time()
            ransac_valid, ransac_rvec, ransac_tvec, relative_ransac_inliers = (
                cv2.solvePnPRansac(
                    np.array(associated_camera_mappoints),
                    np.array(associated_camera_features),
                    self.data_manager.config.cameras[0].K,
                    self.data_manager.config.cameras[0].D,
                    reprojectionError=self.settings["tracking_outlier_reproj_thresh"]
                    ** 2,
                )
            )

            # print("cv PnP: {:.1f} ms".format((time.time()-t_pnp0)*1000))

            # Draw keypoints:
            PnPPose = Pose(ransac_tvec, cv2.Rodrigues(ransac_rvec)[0]).invert()
            uv_kpts, valid = camera.projectPoints(
                self.data_manager.config.gt_kps, None, PnPPose
            )
            self.data_manager.draw_features(
                cam, uv_kpts, (255, 255, 255), size=2, thickness=2
            )

            # Draw RANSAC PnP outliers with white circle
            relative_ransac_inliers = np.squeeze(relative_ransac_inliers)
            outlier_associated_camera_features = []
            # inlier_associated_camera_features = np.array(associated_camera_features)[np.squeeze(relative_ransac_inliers)]

            if len(relative_ransac_inliers) != len(
                associated_camera_features
            ):  # if outliers are present
                next_id = 0
                for i in range(len(associated_camera_features)):
                    if next_id == len(relative_ransac_inliers):
                        break
                    if i == relative_ransac_inliers[next_id]:
                        next_id += 1
                        continue
                    outlier_associated_camera_features.append(
                        associated_camera_features[i]
                    )

                self.data_manager.draw_features(
                    cam,
                    outlier_associated_camera_features,
                    (255, 255, 255),
                    size=15,
                    thickness=2,
                )
                # =================================================

        t1 = time.time()
        prnt_str = "Feature Matching: {:.1f} ms".format((t1 - t0) * 1000)
        print("{} - {} features".format(prnt_str, count))

        # look at distribution of losses:
        # if self.data_manager.data.currentMap.pose_idx == 4:
        #     plt.hist(desc_loss,15)
        #     plt.show()
        #     plt.hist(reproj_loss,15)
        #     plt.show()
        #     sys.exit(0)

        if count < self.settings["tracking_min_matches"]:
            return 0

        #########################################################################

        # Optimization using associated keypoints:
        self.optimizer.PoseOptimization(data)

        #########################################################################

        # Draw optimized pose and filter out outliers:
        for cam in range(self.data_manager.config.nrcams):
            t2 = time.time()
            camera = self.data_manager.config.cameras[cam]

            # Draw:
            uv_kpts, valid = camera.projectPoints(
                self.data_manager.config.gt_kps,
                self.currentMultiframe.T_origin2body_estimated,
            )
            uv_kpts = uv_kpts[valid]
            self.data_manager.draw_features(
                cam, uv_kpts, (255, 0, 0), size=4, thickness=4
            )  # B
            t25 = time.time()

            # MapPoint and features: TODO make it neater (it's decently fast...)
            associated_mappoints: List[MapPoint] = [
                self.data_manager.data.currentMap.mappoints[idx]
                for idx in associated_frame_mappoints_idx[cam]
            ]
            associated_features = np.array(
                [
                    self.currentMultiframe.feature_pts[cam][idx]
                    for idx in associated_frame_features_idx[cam]
                ]
            )
            # Reproject into image
            uv_curr_reproj, valid = camera.projectPoints(
                np.array([mapPoint.cartesian for mapPoint in associated_mappoints]),
                self.currentMultiframe.T_origin2body_estimated,
            )
            t3 = time.time()

            # filter out non-visible map points: NOTE if not reprojectable -> outlier, but we can assume the all map points are in view as they have already been filtered with IMU propagated pose
            # uv_curr_reproj = uv_curr_reproj[valid]
            # associated_features = associated_features[valid]

            # Filter by reprojection error:
            reproj_errors = np.linalg.norm(associated_features - uv_curr_reproj, axis=1)
            outliers = (
                reproj_errors > self.settings["tracking_outlier_reproj_thresh"] ** 2
            )
            inliers = (
                reproj_errors <= self.settings["tracking_outlier_reproj_thresh"] ** 2
            )
            t4 = time.time()

            outlier_mappoints = list(compress(associated_mappoints, outliers))
            # Delete observation TODO: maybe keep tentative observations, and only add to observations if declared inlier... maybe faster
            for outlier_mappoint in outlier_mappoints:
                for i in range(
                    len(outlier_mappoint.observations) - 1, -1, -1
                ):  # propagate backwards for most recent observations:
                    if (
                        outlier_mappoint.observations[i].frame_idx
                        == self.currentMultiframe.frame_idx
                        and outlier_mappoint.observations[i].camera_idx == cam
                    ):
                        del outlier_mappoint.observations[i]
                        break  # one observation per frame per camera
            t5 = time.time()

            # Set feature association:
            inlier_associated_camera_features_idx = np.array(
                associated_frame_features_idx[cam]
            )[inliers]
            for idx in inlier_associated_camera_features_idx:
                self.currentMultiframe.associations[cam][idx] = True
            t6 = time.time()

            # draw outliers:
            self.data_manager.draw_features(
                cam, associated_features[outliers], (255, 0, 0), thickness=5
            )

            t7 = time.time()
            # print("Outlier: {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} - {:.1f} ms".format((t25-t2)*1000, (t3-t25)*1000,(t4-t3)*1000,(t5-t4)*1000,(t6-t5)*1000,(t7-t6)*1000,(t7-t2)*1000))

        return 1

    # =================================== INITIALIZATION METHODS ===================================

    def set_init(self):

        # make sure all images are present. init frame needs to have all images:
        if not any(x is None for x in self.currentMultiframe.imgs):
            # check min number of features:
            max_feaures = 0
            for cam in range(self.data_manager.config.nrcams):
                max_feaures = max(
                    max_feaures, len(self.currentMultiframe.feature_pts[cam])
                )
                self.data_manager.draw_features(
                    cam, self.currentMultiframe.feature_pts[cam], (0, 255, 0)
                )

            if max_feaures > self.settings["min_features"]:
                # Set estimated pose as ground truth
                # self.currentMultiframe.T_origin2body_estimated = self.currentMultiframe.T_origin2body
                # self.currentMultiframe.vel_estimated = self.currentMultiframe.enu_vel
                # self.currentMultiframe.vel_estimated_covariance = np.diag([0.1,0.1,0.1])

                # set identity for the intial frame
                self.currentMultiframe.T_origin2body_estimated = Pose(
                    np.array([0, 0, 0]),
                    np.eye(3),
                    np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                )  # position then rotation
                self.currentMultiframe.vel_estimated = np.array([0, 0, 0]).T
                self.currentMultiframe.vel_estimated_covariance = np.diag(
                    [0.1, 0.1, 0.1]
                )

                # Set as initial frame
                self.initMultiframe = self.currentMultiframe
                return 1
            else:
                print(
                    "[SLAM] (NOT_INITIALIZED) {} detected < {} required features".format(
                        max_feaures, self.settings["min_features"]
                    )
                )
        else:
            print("[SLAM] (NOT_INITIALIZED) Not all images present, cannot intialize")
        return 0

    def initialize(self):

        t0 = time.time()

        # # Set estimated pose as Ground truth:
        # self.currentMultiframe.T_origin2body_estimated = self.currentMultiframe.T_origin2body
        # self.currentMultiframe.vel_estimated = self.currentMultiframe.enu_vel
        # self.currentMultiframe.vel_estimated_covariance = np.diag([0.1,0.1,0.1])

        # set identity for the intial frame
        self.currentMultiframe.T_origin2body_estimated = Pose(
            np.array([0, 0, 0]),
            np.eye(3),
            np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        )  # position then rotation
        self.currentMultiframe.vel_estimated = np.array([0, 0, 0]).T
        self.currentMultiframe.vel_estimated_covariance = np.diag([0.1, 0.1, 0.1])

        # check min number of features:
        max_feaures = 0
        for cam in range(self.data_manager.config.nrcams):
            max_feaures = max(max_feaures, len(self.currentMultiframe.feature_pts[cam]))

        if max_feaures < self.settings["min_features"]:
            print(
                "[SLAM] (INITIALIZING) less than {} features".format(self.min_features)
            )
            return 0

        # Feature matching:
        all_matches = []
        all_xyz_point = []
        tot_matches = 0
        for cam in range(self.data_manager.config.nrcams):
            matcher = cv2.BFMatcher()
            #                       query                                train
            matches = matcher.match(
                self.initMultiframe.descriptors[cam],
                self.currentMultiframe.descriptors[cam],
            )
            matches = sorted(matches, key=lambda x: x.distance)
            # print(matches[70].distance)
            for i in range(len(matches)):
                #     print(matches[i].distance)
                if matches[i].distance > self.settings["init_min_desc_distance"]:
                    break
            del matches[i:]

            all_matches.append(matches)
            tot_matches += len(matches)

        t1 = time.time()
        print("Feature Matching: {:.1f} ms".format((t1 - t0) * 1000))

        if tot_matches < self.settings["init_min_matches"]:
            return 0

        # Keypoint Initialization:
        total_initializations = 0
        self.data_manager.data.currentMap = Map()
        for cam in range(self.data_manager.config.nrcams):
            if self.currentMultiframe.imgs[cam] is not None:

                matches, uv_init, uv_curr, xyz_point = associate_by_triangulation(
                    all_matches[cam],
                    cam,
                    self.initMultiframe,
                    self.currentMultiframe,
                    {
                        "max_reproj_dist": self.settings["max_reproj_dist"],
                        "min_reproj_err": self.settings["init_min_reproj_err"],
                    },
                )

                camera = self.data_manager.config.cameras[cam]

                all_matches[cam] = matches  # list(compress(all_matches[cam], correct))

                # create map points and add to Map:
                draw_features = []
                for i in range(len(xyz_point)):
                    # og_index = og_indexing[i]
                    covariance = np.diag([0.1, 0.1, 0.1])  # TODO
                    ptMap = MapPoint(xyz_point[i], covariance)
                    ptMap.addObservation(
                        cam,
                        self.initMultiframe.frame_idx,
                        uv_init[i],
                        self.initMultiframe.descriptors[cam][
                            all_matches[cam][i].queryIdx
                        ],
                    )
                    ptMap.addObservation(
                        cam,
                        self.currentMultiframe.frame_idx,
                        uv_curr[i],
                        self.currentMultiframe.descriptors[cam][
                            all_matches[cam][i].trainIdx
                        ],
                    )  # TODO maybe change this s.t. MultiFrames have observations instead ...? Or both?
                    self.data_manager.data.currentMap.addMapPoint(ptMap)
                    draw_features.append(
                        self.currentMultiframe.feature_pts[cam][
                            all_matches[cam][i].trainIdx
                        ]
                    )

                total_initializations += len(xyz_point)

                self.data_manager.draw_features(cam, draw_features, (0, 0, 255))

        if total_initializations < 10:  # TODO threshold
            return 0
        # Associate keypoints in different images to each other using distance and descriptors
        # TODO based on approximate base2cams, should be able to filter out which ones are overlapping. Should account for calibration error
        """
        Use orb matcher to match features, then for every matching feature, use distance to filter
        """
        # for this_cam in range(self.data_manager.config.nrcams-1):
        #     for other_cam in range(this_cam+1, self.data_manager.config.nrcams):
        #         for this_match_idx in range(len(all_matches[this_cam])):
        #             for other_match_idx in range(len(all_matches[other_cam])):

        #                 # this_feature = self.initMultiframe.feature_pts[this_cam][all_matches[this_cam][this_match_idx]]
        #                 this_descriptor = self.initMultiframe.descriptors[this_cam][all_matches[this_cam][this_match_idx]]
        #                 other_descriptor = self.initMultiframe.descriptors[other_cam][all_matches[other_cam][other_match_idx]]

        #                 this_pt = all_xyz_point[this_cam][this_match_idx]
        #                 other_pt = all_xyz_point[other_cam][other_match_idx]

        # print("Ground plane projection: {:.1f} ms".format((time.time() - t1) * 1000))

        # if this is only the first frame, then i should return without
        if self.loops == 0:
            self.loops = self.loops + 1
            return 0

        # TODO data sharing through data_manager
        data = {}
        data["MFs"] = [self.initMultiframe, self.currentMultiframe]
        data["map"] = self.data_manager.data.currentMap
        data["accumulated_imu"] = self.data_manager.data.accumulated_imus
        self.optimizer.InitialOptimization(
            data
        )  # optimizes for imu bias and map points

        uv_kpts, valid = camera.projectPoints(
            self.data_manager.config.gt_kps,
            self.currentMultiframe.T_origin2body_estimated,
        )
        self.data_manager.draw_features(cam, uv_kpts, (255, 255, 0), size=5)

        return 1

    def filter_overlapping_features(self, cam0: Camera, cam1: Camera):
        """
        calculates which pixel areas are overlapping
        either by calculating projection vectors of mid-edges and corners of one image, and seeing if it projects into the other image
        should return a mask in both image frames, and should be done once on startup or calibration update
        """
        pass
