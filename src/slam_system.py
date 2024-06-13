#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu


import yaml
import gtsam
from threading import Thread
import multiprocessing
import math
from skimage.measure import ransac

from data_manager import DataManager, SLAMConfig
from models import *
from pose import Pose
from camera import *
from tracker_gt_init import Tracker
from tracker_monocam import *
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
        # self.localMapper = LocalMapper()

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
        # TODO debug my IMU initialization
        # ============ Setup IMU parameters ============
        # imu_params = gtsam.PreintegrationParams.MakeSharedU(
        #     0
        # )  # gravity along negative z axis

        # imu_params.setAccelerometerCovariance(
        #      2.0000e-3 * np.eye(3)
        # )  # acc white noise in continuous

        # imu_params.setIntegrationCovariance(
        #     1e-7 * np. eye(3)
        # )  # integration uncertainty continuous

        # imu_params.setGyroscopeCovariance(
        #     1.6968e-04 * np.eye(3)
        # )  # gyro white noise in continuous
        # # imu_params.setOmegaCoriolis(w_coriolis)
        # self.data_manager.update_config(imu_params=imu_params)

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
        # TODO debug the IMU initialization i set
        # ============ Setup IMU parameters ============
        # imu_params = gtsam.PreintegrationParams.MakeSharedU(
        #     0
        # )  # gravity along negative z axis

        # imu_params.setAccelerometerCovariance(
        #      2.0000e-3 * np.eye(3)
        # )  # acc white noise in continuous

        # imu_params.setIntegrationCovariance(
        #     1e-7 * np. eye(3)
        # )  # integration uncertainty continuous

        # imu_params.setGyroscopeCovariance(
        #     1.6968e-04 * np.eye(3)
        # )  # gyro white noise in continuous

        # # imu_params.setOmegaCoriolis(w_coriolis)
        # self.data_manager.update_config(imu_params=imu_params)

        # ============================== Detectors ==============================

    def process(self, timestamp, imgs, accumulated_imu, interpolated_vals):
        print("\nSLAMSystem: Starting process Loop")

        if self.data_manager.config.reset:
            print("SLAM system resetting, skipping process loop.")
            return None

        ####
        # TODO: Move all of the following into its own monocular tracker class
        ####

        # process the frame:
        current_frame:MultiFrame = MultiFrame(timestamp, imgs, len(self.data_manager.data.MKFs))

        # if this is the first frame then add it and move on
        if len(self.data_manager.data.MKFs) == 0:
            self.data_manager.data.MKFs.append(current_frame)
            return None

        # if this is not the first frame then grab the previous
        prev_frame:MultiFrame = self.data_manager.data.MKFs[-1]

        # MATCH FRAMES - get an fundamenalt matric estimate with RANSAC
        # includes inliers for both the frames
        idx1, idx2, rt = self.matchFrames(current_frame, prev_frame)

        if rt is None:
            # in the event tha RANSAC or the correspondence search failed, then return and dont add this frame
            # bad luck or just a dud frame
            return None

        # if I made it this far then i found correspondences cross frames
        self.data_manager.data.MKFs.append(current_frame)

        # add new observations to the map 
        for i,idx in enumerate(idx2):
            if prev_frame.associations[0][idx] is not None and current_frame.associations[0][idx1[i]] is None:
                prev_frame.associations[0][idx].add_observation(current_frame, idx1[i])

        # if frame.id < 5 or True:
        # # get initial positions from fundamental matrix
        # f1.pose = np.dot(Rt, f2.pose)
        # else:
        # # kinematic model (not used)
        # velocity = np.dot(f2.pose, np.linalg.inv(self.mapp.frames[-3].pose))
        # f1.pose = np.dot(velocity, f2.pose)

        # GET INITIAL POSE Optimization

        # Project all existing map points into the current frame

        # prune points behind camera that can't projct
        # match the good points within the orb distane

        # match points with no matches via triangulation
        # do the triangulation in the global frame

        # add any new points to the map. Make sure the point is in front of both frames otherwise rejext
        # add the points

        # optimize the map- g2o instead of gtsam because i dont get it
        # return the pose
        # measure the time that took

        # run a track loop and return all of the
        # output = self.tracker.track(timestamp, imgs, accumulated_imu, interpolated_vals)

        # try a different tracking function with a different intialization strategy
        return

    def matchFrames(self, curr_frame: MultiFrame, prev_frame: MultiFrame):
        # this class is assuming multiframe contains contains only 1 set of keypoints/descriptors
        # would need to do something different for stereo

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )  # norm hamming recommended for ORB features
        matches = matcher.knnMatch(
            curr_frame.descriptors[0], prev_frame.descriptors[0], k=2
        )

        Matched = cv2.drawMatchesKnn(curr_frame.imgs[0], 
                             curr_frame.keypoints[0], 
                             prev_frame.imgs[0], 
                             prev_frame.keypoints[0], 
                             matches, 
                             outImg=None, 
                             matchColor=(0, 155, 0), 
                             singlePointColor=(0, 255, 255), 
                             flags=0
                             ) 
  
        # Displaying the image  
        cv2.imwrite('Match.jpg', Matched) 

        # use Lowe's ratio test to reduce false matches
        correspondences = []  # store tuples of correspondences
        idx1, idx2 = [], []
        idx1s, idx2s = set(), set()

        for m, n in matches:
            # every match is a tuple of the match in each frame

            if m.distance < 0.75 * n.distance:
                p1 = curr_frame.feature_pts[m.queryIdx]
                p2 = prev_frame.feature_pts[m.trainIdx]

                # be within orb distance 32
                if m.distance < 32:

                    if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                        idx1.append(m.queryIdx)
                        idx2.append(m.trainIdx)
                        idx1s.add(m.queryIdx)
                        idx2s.add(m.trainIdx)
                        correspondences.append((p1, p2))

        # check that i have at least 8 correspondences before estimating F
        if len(correspondences) < 8:
            # if not just return None and handle it on the other end
            return None, None, None

        # setup RANSAC for estimating F with 8-pt algo
        ransac_residual_threshold = 0.2
        ransac_max_trials = 100
        corr_np = np.array(correspondences)
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)

        # make the pooints homogenous

        # TODO: something in my fundamental matrix model is not stable
        # corr_np = np.append(corr_np, np.ones((corr_np.shape[0], 2, 1)), axis=2)
        model, inliers = ransac(
            (corr_np[:, 0], corr_np[:, 1]),
            FundamentalMatrixTransform,
            min_samples=8,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials,
        )
        print(
            "Matches:  %d -> %d -> %d -> %d"
            % (len(curr_frame.descriptors[0]), len(matches), len(inliers), sum(inliers))
        )

        rt = fundamentalToRt(model.params)

        # get a rotation and translation from the fundamental matrix
        # print(model.params)

        # intrinsics = self.data_manager.config.cameras[0].K_

        # E = intrinsics.T @ (
        #     model.params @ (intrinsics)
        # )
        # rt = estimate_RT_from_E(E,corr_np,intrinsics)

        return idx1[inliers], idx2[inliers], rt

    # TODO
    def updateCalibration(self, calibrations):
        """
        Change the calibration system wide:
        - self.data_manager
        - map and poses in factor graphs

        """
        pass


def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret


# pose
def fundamentalToRt(F):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]

    # TODO: Resolve ambiguities in better ways. This is wrong.
    if t[2] < 0:
        t *= -1

    return np.linalg.inv(poseRt(R, t))


def estimate_initial_RT(E):
    # perform SVD on E

    u, s, vt = np.linalg.svd(E)

    #  use E = MQ where M = UZUt and Q = UWVt
    # set values for Z and W

    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # compute M and both guess for Q
    M = u.dot(Z).dot(u.T)
    Q1 = u.dot(W).dot(vt)
    Q2 = u.dot(W.T).dot(vt)

    # now i can compute R = det(Q)Q for each guess of Q
    R1 = np.linalg.det(Q1) * Q1
    R2 = np.linalg.det(Q2) * Q2

    # there are two guesses for T, either u3 or -u3 with u3  being the third column vector of u
    tvec1 = u[:, 2]
    tvec2 = tvec1 * -1

    # create a tensor with each combination of RT
    rt = np.zeros((4, 3, 4))
    rt[0] = np.hstack((R1, tvec1.reshape((3, 1))))
    rt[1] = np.hstack((R1, tvec2.reshape((3, 1))))
    rt[2] = np.hstack((R2, tvec1.reshape((3, 1))))
    rt[3] = np.hstack((R2, tvec2.reshape((3, 1))))

    return rt

def linear_estimate_3d_point(image_points, camera_matrices):
    # solve for the 3D point by doing AP=0
    # A is formed by a set of similarity transforms
    A = np.ndarray(0)
    # loop through the points to populate A
    for idx, pt in enumerate(image_points):
        A = np.append(A, pt[0] * camera_matrices[idx][2] - camera_matrices[idx][0])
        A = np.append(A, pt[1] * camera_matrices[idx][2] - camera_matrices[idx][1])

    A = A.reshape(image_points.shape[0] * 2, 4)

    # solve with SVD and get the null space
    u, s, vt = np.linalg.svd(A)

    # convert the point from homogenous to 3d euclidean
    pt_3d = vt[-1, :3] / vt[-1, -1]
    return pt_3d


def jacobian(point_3d, camera_matrices):
    # form the jacobian by taking the partial derivatices of the reprojection error
    # loop through the camera matrices
    J = np.ndarray(0)

    # make 3d point homogenous
    homogenous_pt_3d = np.append(point_3d, 1)

    for idx, mtx in enumerate(camera_matrices):
        # project the point
        pt = mtx.dot(homogenous_pt_3d.T)

        # dont normalize since the third value is used in the jacobian
        # normalize the point
        # pt = pt / pt[-1]

        # form the jacobian for this row
        # start with jacobian for X
        for iidx, val in enumerate(pt[:2]):
            deriv_P1 = (pt[2] * mtx[iidx][0] - val * mtx[2][0]) / (pt[2] ** 2)
            deriv_P2 = (pt[2] * mtx[iidx][1] - val * mtx[2][1]) / (pt[2] ** 2)
            deriv_P3 = (pt[2] * mtx[iidx][2] - val * mtx[2][2]) / (pt[2] ** 2)

            J = np.append(J, [deriv_P1, deriv_P2, deriv_P3])

    return J.reshape(2 * camera_matrices.shape[0], 3)

def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # Use Gauss-Newton to estimate the nonlinear 3d point
    # i need an initial estimate  to initalize the method and I can use the linear estimate for this
    p_est = linear_estimate_3d_point(image_points, camera_matrices)

    # run the optimizer for 10 iterations only per the ps2 pdf.
    num_iter = 10
    for i in range(num_iter):
        ## get jacobian and reprojection error vector for each new estimate
        J = jacobian(p_est, camera_matrices)
        e = reprojection_error(p_est, image_points, camera_matrices)

        # update point estimate
        loss = J.T.dot(J)
        loss = np.linalg.inv(loss)
        loss = loss.dot(J.T)
        loss = loss.dot(e.reshape((e.shape[0], 1)))

        p_est = p_est - loss.T

    return p_est

def reprojection_error(point_3d, image_points, camera_matrices):
    # first compute the project points
    reproj_vector = np.ndarray(0)

    # make 3d point homogenous
    homogenous_pt_3d = np.append(point_3d, 1)

    # loop through each image points to calculate the reprojection error from the 3D point to the point in image space
    for idx, pt in enumerate(image_points):
        # reproject the 3d point
        proj_pt = camera_matrices[idx].dot(homogenous_pt_3d)
        # get rid of the homegenous part  of the projected point
        proj_pt = proj_pt[:2] / proj_pt[-1]

        err = proj_pt - pt[:2]
        reproj_vector = np.append(reproj_vector, err[0])
        reproj_vector = np.append(reproj_vector, err[1])

    return reproj_vector

def estimate_RT_from_E(E, image_points, K):

    potential_rt = estimate_initial_RT(E)
    rt_points_3d = np.zeros((potential_rt.shape[0], image_points.shape[0], 3))
    num_postivie_depth_pts = np.zeros(4)

    # loop through each of the possible RT's
    for idx, rt in enumerate(potential_rt):
        # make the camera matrices for each image
        # M = k.dot(rt)
        camera_matrices = np.zeros((2, 3, 4))
        camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
        camera_matrices[1, :, :] = K.dot(rt)

        # get the nonlinear estimate for the 3D point of the measured points
        for iidx, meas in enumerate(image_points):
            rt_points_3d[idx, iidx, :] = nonlinear_estimate_3d_point(
                meas, camera_matrices
            )[0]

            # while I'm here do a depth check in both frames
            # add a row to the rotation
            rt_homogenous = np.vstack((rt, [0, 0, 0, 1]))
            pt3d_f1 = np.append(rt_points_3d[idx, iidx, :], 1)
            pt3d_f2 = rt_homogenous.dot(pt3d_f1)

            if pt3d_f1[2] > 0 and pt3d_f2[2] > 0:
                num_postivie_depth_pts[idx] = num_postivie_depth_pts[idx] + 1

    return potential_rt[num_postivie_depth_pts.argmax()]
