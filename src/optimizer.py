from __future__ import print_function
import sys
from pathlib import Path

from joblib import PrintTime
import cv2
from gtsam import symbol_shorthand
import copy
from typing import List, Optional
from functools import partial

from pose import Pose
from imu import IMU
from mapping import *
from multiframe import MultiFrame
from camera import *
from data_manager import DataManager

B = symbol_shorthand.B # Bias
V = symbol_shorthand.V # Velocity
X = symbol_shorthand.X # Imu Pose
P = symbol_shorthand.P # Points
K = symbol_shorthand.K # Intrinsics
Y = symbol_shorthand.Y # Camera cal
C = symbol_shorthand.C # Camera pose

import numpy as np
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
import gtsam
import time

import inspect

# pose between factor:
def error_cal_between(measurement: np.ndarray, this: gtsam.CustomFactor,
                      values: gtsam.Values,
                      jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """
    Extrinsic calibration between factor
    relative rotation change:
    """
    x0 = this.keys()[0]# X(n-1)
    x1 = this.keys()[1]# X(n)
    xc = this.keys()[2]# C(i)

    zero_tf = values.atPose3(x0).transformPoseTo(values.atPose3(x1)).transformPoseTo(values.atPose3(xc))

    # rvec_diff = zero_tf.rotation().xyz()
    # tvec_diff = zero_tf.translation()

    # error = np.concatenate((rvec_diff, tvec_diff))
    error = gtsam.Pose3.Logmap(zero_tf)

    J_diff1 = np.eye(6)#np.zeros((6,6))
    # J_diff1[:3,:3] = np.eye(3)

    if jacobians is not None:
        jacobians[0] = -J_diff1
        jacobians[1] = J_diff1
        jacobians[2] = J_diff1

    return error

def error_plane(measurement: np.ndarray, this: gtsam.CustomFactor,
                values: gtsam.Values,
                jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    
    point = values.atPoint3(this.keys()[0])
    error = measurement - point[2]

    if jacobians is not None:
        J_height = np.zeros((1,6))
        J_height[0][2] = -1
        jacobians[0] = J_height

    return error
    


class Optimizer:

    def __init__(self) -> None:
        self.data_manager = DataManager()
        self.sigma_b = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0)

        """
         GTSAM COVARIANCE IS ROTATION,TRANSLATION
         POSE COVARIANCE IS TRANSLATION,ROTATION
        """

    

    def InitialOptimization(self, data):
        """
        Uses ground truth pose to estimate:
        IMU bias
        Refine keypoints
        """

        

        t0 = time.time()
        

        # TODO change inputting
        # points:
        points: List[MapPoint] = copy.deepcopy(data['map'].mappoints)
        # Poses:
        frames: List[MultiFrame] = data['MFs'] # should be of length 2
        # IMU:
        accumulated_imus: List[IMU] = data['accumulated_imu']

        # Setup initial values to be loaded into map:
        factor_graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        fg_ = gtsam.NonlinearFactorGraph()
        pose_idx = 0

        # ============ Adding everything: ==============

        # ----------- Calibration Priors: -----------
        self.add_cameras(factor_graph, values, intrinsic_noise=gtsam.noiseModel.Isotropic.Sigmas(np.array([100,100,0,0,0,0,0,0,0])), extrinsic_noise=gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0017,0.0017,0.0017,0,0,0])))
        self.add_cameras(fg_, None, intrinsic_noise=gtsam.noiseModel.Isotropic.Sigmas(np.array([100,100,0,5,5,0,0,0,0])), extrinsic_noise=gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0017,0.0017,0.0017,0,0,0])))

        # ============ Setup prior on Pose 0: ============
        self.add_pose(factor_graph, values, pose_idx, self.pose2gtsam(frames[pose_idx].T_origin2body_estimated), gtsam.noiseModel.Diagonal.Sigmas(np.array([0,0,0,0,0,0])))
        self.add_pose(fg_, None, pose_idx, self.pose2gtsam(frames[pose_idx].T_origin2body_estimated), gtsam.noiseModel.Diagonal.Sigmas(np.array([0,0,0,0,0,0])),True)
        vel_xn = frames[pose_idx].vel_estimated


        factor_graph.addPriorVector(V(pose_idx), vel_xn, gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*0.1))# 0.1 ms-1
        factor_graph.addPriorConstantBias(B(pose_idx), gtsam.imuBias.ConstantBias(),gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.01))
        fg_.addPriorVector(V(pose_idx), vel_xn, gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*0.1))# 0.1 ms-1
        fg_.addPriorConstantBias(B(pose_idx), gtsam.imuBias.ConstantBias(),gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.01))
        values.insert(V(pose_idx), vel_xn)
        values.insert(B(pose_idx), gtsam.imuBias.ConstantBias())# zero bias

        pose_idx += 1

        # ============ Setup prior and IMU on Pose 1: ============

        for accumulated_imu in accumulated_imus:
            if accumulated_imu.frame_id_corr == frames[pose_idx].frame_idx:

                self.add_imu(factor_graph,values,pose_idx,accumulated_imu.summed_imu_z)
                self.add_imu(fg_,values,pose_idx,accumulated_imu.summed_imu_z)
        
        # IMU and camera priors:

        self.add_pose(factor_graph, values, pose_idx, self.pose2gtsam(frames[pose_idx].T_origin2body_estimated), gtsam.noiseModel.Diagonal.Sigmas(np.array([0,0,0,0,0,0])))
        self.add_pose(fg_, None, pose_idx, self.pose2gtsam(frames[pose_idx].T_origin2body_estimated), gtsam.noiseModel.Diagonal.Sigmas(np.array([0,0,0,0,0,0])),True)

        values.insert(V(pose_idx), frames[pose_idx].vel_estimated)
        values.insert(B(pose_idx), gtsam.imuBias.ConstantBias())

        # ------------ Keypoint Priors: ------------
        for pt in range(len(points)):
            # Initialize with high uncertainty:
            cartesian_point = gtsam.Point3(points[pt].cartesian[0],points[pt].cartesian[1],points[pt].cartesian[2]).T#xyz
            factor_graph.push_back(gtsam.PriorFactorPoint3(
                P(points[pt].id),
                cartesian_point,
                gtsam.noiseModel.Diagonal.Sigmas(np.array([100,100,100]))
            ))
            fg_.push_back(gtsam.PriorFactorPoint3(
                P(points[pt].id),
                cartesian_point,
                gtsam.noiseModel.Diagonal.Sigmas(np.array([100,100,100]))
            ))
            values.insert(P(points[pt].id), cartesian_point)

        
        # ----------- Detected keypoints: -----------
        frame_idx_map = [frames[i].frame_idx for i in range(len(frames))]
        for pt in range(len(points)):
            for observation in points[pt].observations:
                frame_idx = frame_idx_map.index(observation.frame_idx)
                camera_factor_idx = frame_idx*self.data_manager.config.nrcams + observation.camera_idx
                uv = gtsam.Point2(observation.uv[0],observation.uv[1])
                noise = gtsam.noiseModel.Isotropic.Sigmas((np.array([5,5])))# eyeballing approx 5 pixels feature uncertianty
                model = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                    noise
                )
                factor_graph.push_back(gtsam.GeneralSFMFactor2Cal3Fisheye(
                    uv,
                    model,
                    C(camera_factor_idx),
                    P(points[pt].id),
                    K(observation.camera_idx)
                ))
                fg_.push_back(gtsam.GeneralSFMFactor2Cal3Fisheye(
                    uv,
                    model,
                    C(camera_factor_idx),
                    P(points[pt].id),
                    K(observation.camera_idx)
                ))
        

        # ========================= Optimization =========================
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        lm = gtsam.LevenbergMarquardtOptimizer(factor_graph, values, params)
        result = lm.optimize()
        marginals = gtsam.Marginals(fg_, result)

        t1 = time.time()
        print("InitialOptimization: {:.1f} ms".format((t1-t0)*1000))


        # ========================= Update info =========================

        # MapPoints
        for pt in range(len(points)):
            data['map'].mappoints[pt].cartesian = result.atPoint3(P(points[pt].id))
            data['map'].mappoints[pt].covariance = marginals.marginalCovariance(P(points[pt].id))
        
        # IMU bias:
        for i in range(len(frames)):
            #TODO should be placed in map instead
            frames[i].imu_bias = result.atConstantBias(B(i)).vector()
            frames[i].imu_bias_covariance = marginals.marginalCovariance(B(i))
        
        # Updating map
        # data['map'].factor_graph = factor_graph
        # data['map'].values = values
        # data['map'].pose_idx = pose_idx
        # data['map'].factor_idx = factor_count

        # Resetting isam2:
        # isam_params = gtsam.ISAM2Params()
        # isam_params.setFactorization("CHOLESKY")
        # isam_params.relinearizeSkip = 10
        # self.isam_optimizer = gtsam.ISAM2(isam_params)

        # self.isam_optimizer.update(factor_graph, values)
        






    def PosePredict(self,data):#, last_frame: MultiFrame, current_frame: MultiFrame):
        """
        
        """

        t0 = time.time()
        

        # TODO change inputting

        # Frames:
        currentMultiframe: MultiFrame = data['currentMultiframe']
        lastMultiframe: MultiFrame = data['lastMultiframe']
        # IMU:
        accumulated_imus: List[IMU] = data['accumulated_imus']

        if data['map'] is None:
            points = []
        else:
            pass #TODO

        # Setup initial values to be loaded into map:
        factor_graph = gtsam.NonlinearFactorGraph()
        fg_ = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        pose_idx = 0

        # ============ Adding everything: ==============
        
        # ----------- Calibration Priors: ----------- (fixed)
        self.add_cameras(factor_graph, values, intrinsic_noise=gtsam.noiseModel.Isotropic.Sigmas(np.array([0,0,0,0,0,0,0,0,0])), extrinsic_noise=gtsam.noiseModel.Diagonal.Sigmas(np.array([0,0,0,0,0,0])))
        self.add_cameras(fg_, None, intrinsic_noise=gtsam.noiseModel.Isotropic.Sigmas(np.array([100,100,0,0,0,0,0,0,0])), extrinsic_noise=gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0017,0.0017,0.0017,0,0,0])))

        
        # Pose 0: ----------------------------------------------------
        self.add_pose(factor_graph, values, pose_idx, self.pose2gtsam(lastMultiframe.T_origin2body_estimated), gtsam.noiseModel.Gaussian.Covariance(lastMultiframe.T_origin2body_estimated.transposedCovariance()))
        self.add_pose(fg_, None, pose_idx, self.pose2gtsam(lastMultiframe.T_origin2body_estimated), gtsam.noiseModel.Gaussian.Covariance(lastMultiframe.T_origin2body_estimated.transposedCovariance()),True)

        # add initial velocity and bias for imu
        factor_graph.addPriorVector(V(0), lastMultiframe.vel_estimated, gtsam.noiseModel.Gaussian.Covariance(lastMultiframe.imu_bias_covariance))
        factor_graph.addPriorConstantBias(B(0), gtsam.imuBias.ConstantBias(lastMultiframe.imu_bias[:3],lastMultiframe.imu_bias[3:]),gtsam.noiseModel.Gaussian.Covariance(lastMultiframe.imu_bias_covariance))
        fg_.addPriorVector(V(0), lastMultiframe.vel_estimated, gtsam.noiseModel.Gaussian.Covariance(lastMultiframe.vel_estimated_covariance))
        fg_.addPriorConstantBias(B(0), gtsam.imuBias.ConstantBias(lastMultiframe.imu_bias[:3],lastMultiframe.imu_bias[3:]),gtsam.noiseModel.Gaussian.Covariance(lastMultiframe.imu_bias_covariance))
        values.insert(V(0), lastMultiframe.vel_estimated)
        values.insert(B(0), gtsam.imuBias.ConstantBias(lastMultiframe.imu_bias[:3],lastMultiframe.imu_bias[3:]))

        # Pose 1: ----------------------------------------------------
        pose_idx = 1

        # IMU: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for accumulated_imu in accumulated_imus:
            if accumulated_imu.frame_id_corr == currentMultiframe.frame_idx:
                
                self.add_imu(factor_graph,values,pose_idx,accumulated_imu.summed_imu_z)
                self.add_imu(fg_,None,pose_idx,accumulated_imu.summed_imu_z)

        values.insert(V(pose_idx), currentMultiframe.vel_estimated)
        values.insert(B(pose_idx), gtsam.imuBias.ConstantBias(lastMultiframe.imu_bias[:3],lastMultiframe.imu_bias[3:]))# Use last frame bias

        # pose - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self.add_pose(factor_graph, values, pose_idx, self.pose2gtsam(currentMultiframe.T_origin2body_estimated), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,100,100,100])))
        self.add_pose(fg_, None, pose_idx, self.pose2gtsam(currentMultiframe.T_origin2body_estimated), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,100,100,100])),True)

        # ========================= Optimization =========================
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        lm = gtsam.LevenbergMarquardtOptimizer(factor_graph, values, params)
        result = lm.optimize()
        marginals = gtsam.Marginals(fg_, result)


        isam_params = gtsam.ISAM2Params()
        isam_params.setFactorization("CHOLESKY")
        isam_params.relinearizeSkip = 10
        isam_optimizer = gtsam.ISAM2(isam_params)
        isam_optimizer.update(factor_graph, values)
        # result = isam_optimizer.calculateEstimate()

        t1 = time.time()
        # print("PosePredict: {:.1f} ms".format((t1-t0)*1000))

        # ========================= Update Values =========================
        # Current Pose:
        currentMultiframe.T_origin2body_estimated = self.gtsam2pose(result.atPose3(X(pose_idx)))
        currentMultiframe.T_origin2body_estimated.setFromTransposedCovariance(marginals.marginalCovariance(X(pose_idx)))# isam_optimizer  marginals
        currentMultiframe.vel_estimated = result.atVector(V(pose_idx))
        currentMultiframe.vel_estimated_covariance = marginals.marginalCovariance(V(pose_idx))

        # IMU bias:
        #TODO should be placed in map instead
        currentMultiframe.imu_bias = result.atConstantBias(B(pose_idx)).vector()
        currentMultiframe.imu_bias_covariance = marginals.marginalCovariance(B(pose_idx))

        return isam_optimizer, pose_idx, factor_graph, values, fg_






    def PoseOptimization(self, data):
        """
        Will only optimize for Pose, and keep calibration/keypoints constant
        """

        t0 = time.time()
        

        # TODO change inputting
        # New frames and map points:
        currentMultiframe: MultiFrame = data['currentMultiframe']
        points: List[MapPoint] = data['map'].mappoints
        
        # extra factors
        extra_factors = data['factor_graph']#gtsam.NonlinearFactorGraph()
        fg_ = data['fg_']#gtsam.NonlinearFactorGraph()
        extra_values = data['values']#gtsam.Values()
        # extra_factors = gtsam.NonlinearFactorGraph()
        # extra_values = gtsam.Values()
        pose_idx: int = data['pose_idx']
        
        # ============ Adding observation from only current frame: ==============

        
        for pt in range(len(points)):
            for observation in points[pt].observations:
                if observation.frame_idx == currentMultiframe.frame_idx:

                    # ------------ Keypoint Priors: ------------
                    cartesian_point = gtsam.Point3(*points[pt].cartesian)#xyz
                    extra_factors.push_back(gtsam.PriorFactorPoint3(
                        P(points[pt].id),
                        cartesian_point,
                        gtsam.noiseModel.Gaussian.Covariance(points[pt].covariance)
                    ))
                    fg_.push_back(gtsam.PriorFactorPoint3(
                        P(points[pt].id),
                        cartesian_point,
                        gtsam.noiseModel.Gaussian.Covariance(points[pt].covariance)
                    ))
                    extra_values.insert(P(pt), cartesian_point)

                    # ----------- Detected keypoints: -----------
                    camera_factor_idx = pose_idx*self.data_manager.config.nrcams + observation.camera_idx
                    uv = gtsam.Point2(*observation.uv)
                    noise = gtsam.noiseModel.Isotropic.Sigmas((np.array([5,5])))
                    model = gtsam.noiseModel.Robust.Create(
                        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                        noise
                    )
                    extra_factors.push_back(gtsam.GeneralSFMFactor2Cal3Fisheye(
                        uv,
                        model,
                        C(camera_factor_idx),
                        P(points[pt].id),
                        K(observation.camera_idx)
                    ))
                    fg_.push_back(gtsam.GeneralSFMFactor2Cal3Fisheye(
                        uv,
                        model,
                        C(camera_factor_idx),
                        P(points[pt].id),
                        K(observation.camera_idx)
                    ))

                    break
        


        # ========================= Optimization =========================
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        lm = gtsam.LevenbergMarquardtOptimizer(extra_factors, extra_values, params)
        result = lm.optimize()
        marginals = gtsam.Marginals(fg_, result)

        # init_isam: gtsam.ISAM2 = data['init_isam']
        # init_isam.update(extra_factors,extra_values)
        # result = init_isam.calculateEstimate()
        print("PoseOptimization: {:.1f} ms".format((time.time()-t0)*1000))

        # ========================= Update Values =========================
        # Current Pose:
        currentMultiframe.T_origin2body_estimated = self.gtsam2pose(result.atPose3(X(pose_idx)))
        currentMultiframe.T_origin2body_estimated.setFromTransposedCovariance(marginals.marginalCovariance(X(pose_idx)))
        currentMultiframe.vel_estimated = result.atVector(V(pose_idx))
        currentMultiframe.vel_estimated_covariance = marginals.marginalCovariance(V(pose_idx))




    def BundleAdjustment(self, data):
        """
        Full BA on everything
        """

        t0 = time.time()
        

        # New frames and map points:
        new_frames: List[MultiFrame] = data['new_frames']
        new_points: List[MapPoint] = data['new_mappoints']

        # IMU:
        accumulated_imus: List[IMU] = self.data_manager.data.accumulated_imus

        factor_graph = self.data_manager.data.currentMap.factor_graph
        values = self.data_manager.data.currentMap.values
        pose_idx = self.data_manager.data.currentMap.pose_idx
        pose_frame_corr = {}

        if len(self.data_manager.data.MKFs) > 0:
            last_frame_idx = self.data_manager.data.MKFs[-1].frame_idx
            t_prev_imu = self.data_manager.data.MKFs[-1].timestamp
        else:
            self.add_cameras(factor_graph, values, intrinsic_noise=gtsam.noiseModel.Isotropic.Sigmas(np.array([100,100,0,0,0,0,0,0,0])), extrinsic_noise=gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0017,0.0017,0.0017,0,0,0])))


        # factor_count = 0
        # prev_factors = []

        # # ============ Adding everything: ==============
        
        for frame in new_frames:
            self.add_pose(factor_graph, values, pose_idx, self.pose2gtsam(frame.T_origin2body_estimated), gtsam.noiseModel.Gaussian.Covariance(frame.T_origin2body_estimated.transposedCovariance()))

            if pose_idx == 0:
                # add initial velocity and bias for imu
                factor_graph.addPriorVector(V(0), frame.vel_estimated, gtsam.noiseModel.Gaussian.Covariance(frame.vel_estimated_covariance))
                factor_graph.addPriorConstantBias(B(0), gtsam.imuBias.ConstantBias(frame.imu_bias[:3],frame.imu_bias[3:]),gtsam.noiseModel.Gaussian.Covariance(frame.imu_bias_covariance))
                last_frame_idx = frame.frame_idx
                t_prev_imu = frame.timestamp
            else:
                # IMU:
                summed_imu_z = gtsam.PreintegratedImuMeasurements(self.data_manager.config.imu_params, gtsam.imuBias.ConstantBias())
                for accumulated_imu in accumulated_imus:
                    # TODO simplify search
                    if accumulated_imu.frame_id_corr > last_frame_idx and accumulated_imu.frame_id_corr <= frame.frame_idx:
                        for j in range(len(accumulated_imu.data)):
                            # Calculate dt:
                            time_imu = accumulated_imu.data[j][0]
                            dt_imu = time_imu - t_prev_imu
                            t_prev_imu = time_imu

                            summed_imu_z.integrateMeasurement(
                                accumulated_imu.data[j][1][3:],
                                accumulated_imu.data[j][1][:3],
                                dt_imu
                            )
                    
                    # integrate last imu measurement:
                    if accumulated_imu.frame_id_corr == frame.frame_idx:
                        summed_imu_z.integrateMeasurement(
                            accumulated_imu.data[-1][1][3:],
                            accumulated_imu.data[-1][1][:3],
                            frame.timestamp - accumulated_imu.data[-1][0]
                        )
                self.add_imu(factor_graph,values,pose_idx,summed_imu_z)
            
            values.insert(V(pose_idx), frame.vel_estimated)
            values.insert(B(pose_idx), gtsam.imuBias.ConstantBias(frame.imu_bias[:3],frame.imu_bias[3:]))

            last_frame_idx = frame.frame_idx
            pose_frame_corr.update({frame.frame_idx:pose_idx})
            pose_idx += 1
        
        # Add observations from tracking existing mappoints:
        if len(self.data_manager.data.MKFs) > 0:
            for mappoint in self.data_manager.data.currentMap.mappoints:
                for observation in mappoint.observations:
                    if observation.frame_idx in pose_frame_corr.keys():
                        # add observations:
                        camera_factor_idx = pose_frame_corr[observation.frame_idx]*self.data_manager.config.nrcams + observation.camera_idx
                        uv = gtsam.Point2(*observation.uv)
                        noise = gtsam.noiseModel.Isotropic.Sigmas((np.array([5,5])))
                        model = gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                            noise
                        )
                        factor_graph.push_back(gtsam.GeneralSFMFactor2Cal3Fisheye(
                            uv,
                            model,
                            C(camera_factor_idx),
                            P(mappoint.id),
                            K(observation.camera_idx)
                        ))
        
        # Add new map points:
        for pt in range(len(new_points)):
            # ------------ Keypoint Priors: ------------
            cartesian_point = gtsam.Point3(*new_points[pt].cartesian)#xyz
            factor_graph.push_back(gtsam.PriorFactorPoint3(
                P(new_points[pt].id),
                cartesian_point,
                gtsam.noiseModel.Gaussian.Covariance(new_points[pt].covariance)
            ))
            values.insert(P(new_points[pt].id), cartesian_point)
            
            # ----------- Detected keypoints: -----------
            for observation in new_points[pt].observations:
                try:
                    camera_factor_idx = pose_frame_corr[observation.frame_idx]*self.data_manager.config.nrcams + observation.camera_idx
                    uv = gtsam.Point2(*observation.uv)
                    noise = gtsam.noiseModel.Isotropic.Sigmas((np.array([5,5])))
                    model = gtsam.noiseModel.Robust.Create(
                        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                        noise
                    )
                    factor_graph.push_back(gtsam.GeneralSFMFactor2Cal3Fisheye(
                        uv,
                        model,
                        C(camera_factor_idx),
                        P(new_points[pt].id),
                        K(observation.camera_idx)
                    ))
                except KeyError:
                    pass


        self.data_manager.data.currentMap.factor_graph = factor_graph
        self.data_manager.data.currentMap.values = values
        self.data_manager.data.currentMap.pose_idx = pose_idx
        t01 = time.time()

        # params = gtsam.LevenbergMarquardtParams()
        # params.setVerbosityLM("ERROR")
        # lm = gtsam.LevenbergMarquardtOptimizer(factor_graph, values, params)
        # result = lm.optimize()
        # time.sleep(1)
        marginals = gtsam.Marginals(factor_graph, values)

        t1 = time.time()
        print("                                         optimiz: {:.1f} ms".format((t1-t01)*1000))
        print("                                         Mapping: {:.1f} ms".format((t1-t0)*1000))
        print("                                         pose IDX: {}".format(pose_idx))

        # Updating:
        currentMapPoints_cp = copy.deepcopy(self.data_manager.data.currentMap.mappoints)
        
        

        if len(self.data_manager.data.MKFs) > 0:
            for mappoint in currentMapPoints_cp:
                mappoint.cartesian = values.atPoint3(P(mappoint.id))
                mappoint.covariance = marginals.marginalCovariance(P(mappoint.id))
            
            for pt in range(len(new_points)):
                new_points[pt].cartesian = values.atPoint3(P(new_points[pt].id))
                new_points[pt].covariance = marginals.marginalCovariance(P(new_points[pt].id))

            t2 = time.time()
            self.data_manager.data.currentMap.mutex.acquire()
            print("                                         A")
            self.data_manager.data.currentMap.mappoints = currentMapPoints_cp
            self.data_manager.data.currentMap.addMapPoints(new_points)
            print("                                         R")
            self.data_manager.data.currentMap.mutex.release()
            print("                                         Done updating: {:.1f} ms (took {:.1f})".format((time.time()-t1)*1000, (time.time()-t2)*1000))







    #     # ============ Adding everything: ==============
        
    #     # ----------- Calibration Priors: -----------
    #     for cam in range(self.data_manager.config.nrcams):
    #         # Intrinsics:
    #         camera: PinholeCamera = self.data_manager.config.cameras[cam]
    #         init_cal = gtsam.Cal3Fisheye(camera.K_[0][0], camera.K_[1][1],0,camera.K_[0][2],camera.K_[1][2],camera.D4[0], camera.D4[1],camera.D4[2],camera.D4[3])
    #         factor_graph.push_back(gtsam.PriorFactorCal3Fisheye(
    #             K(cam),
    #             init_cal,
    #             gtsam.noiseModel.Isotropic.Sigmas(np.array([0,0,0,0,0,0,0,0,0]))
    #         ))
    #         factor_count += 1
    #         values.insert(K(cam), init_cal)
        
    #         # Extrinsics:
    #         extrinsic_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0,0,0,0,0,0]))
    #         extrinsic = self.pose2gtsam(camera.base2cam)
    #         factor_graph.addPriorPose3(Y(cam), extrinsic, extrinsic_noise)
    #         values.insert(Y(cam), extrinsic)
    #         factor_count += 1


    #     # ============ Loop through poses to add priors and IMU ============
    #     # TODO: frame_idx and camera_idx to factor idx mapping:
    #     # and NOTE that in this case, self.factor_idx == i below in the for loop
    #     frame_idx_map = [frames[i].frame_idx for i in range(len(frames))]
    #     pose_idx = 0
    #     for i in range(len(frames)):
    #         pose_xn = self.pose2gtsam(frames[i].T_origin2body)
    #         vel_xn = frames[i].enu_vel
            
    #         frame_srt_fx = factor_count
    #         if pose_idx==0:
    #             # ============ Setup prior: ============
    #             values.insert(X(0), pose_xn)
    #             values.insert(V(0), vel_xn)
    #             values.insert(B(0), gtsam.imuBias.ConstantBias())# zero bias
                
    #             prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0017,0.0017,0.0017,0.01,0.01,0.01]))# 0.1 DEGREES    10 CM
    #             # print(factor_count)
    #             factor_graph.addPriorPose3(X(0), pose_xn, prior_pose_noise)
    #             factor_graph.addPriorVector(V(0), vel_xn, gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*0.1))# 0.1 ms-1
    #             factor_graph.addPriorConstantBias(B(0), gtsam.imuBias.ConstantBias(),self.sigma_b)
    #             prev_factors.extend([*range(factor_count,factor_count+3)])
    #             factor_count += 3
    #         else:
    #             srt_rm = factor_count
    #             for accumulated_imu in accumulated_imus:
    #                 if accumulated_imu.frame_id_corr == frames[i].frame_idx:
                
    #                     # Create IMU factor
    #                     factor_graph.push_back(gtsam.ImuFactor(
    #                         X(pose_idx-1), V(pose_idx-1),
    #                         X(pose_idx), V(pose_idx),
    #                         B(pose_idx-1),
    #                         accumulated_imu.summed_imu_z
    #                     ))

    #                     # Bias between: (constant)
    #                     factor_graph.push_back(gtsam.BetweenFactorConstantBias(
    #                         B(pose_idx-1),
    #                         B(pose_idx),
    #                         gtsam.imuBias.ConstantBias(),
    #                         self.sigma_b
    #                     ))
    #                     factor_count += 2
    #             prev_factors.extend([*range(srt_rm,factor_count+3)])
                
    #             # IMU and camera priors:
    #             imu_pose_prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0017,0.0017,0.0017,0.01,0.01,0.01]))# 0.1 DEGREES    10 CM
    #             factor_graph.addPriorPose3(X(pose_idx), pose_xn, imu_pose_prior)
    #             values.insert(X(pose_idx), pose_xn)
    #             values.insert(V(pose_idx), vel_xn)
    #             values.insert(B(pose_idx), gtsam.imuBias.ConstantBias())
    #             factor_count += 1
            
    #         frames[i].factors.extend([*range(frame_srt_fx,factor_count)])

    #         srt_rm = factor_count
    #         # Add imu2cam extrinsic between factors
    #         for cam in range(self.data_manager.config.nrcams):
    #             camera: PinholeCamera = self.data_manager.config.cameras[cam]

    #             # IMU to camera prior using custom factor:
    #             prior_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0,0,0,0,0,0]))
    #             cam_idx = pose_idx*self.data_manager.config.nrcams + cam
    #             factor_graph.add(gtsam.CustomFactor(prior_between_noise, [X(pose_idx), C(cam_idx), Y(cam)],partial(error_cal_between, [None])))
    #             values.insert(C(cam_idx), self.pose2gtsam(frames[i].T_origin2body.convert(camera.base2cam)))
    #             factor_count += 1
            
    #         pose_idx += 1
    #         if pose_idx == 0:
    #             prev_factors.extend([*range(srt_rm,factor_count+3)])

    #     # ------------ Keypoint Priors: ------------
    #     for pt in range(len(points)):
    #         cartesian_point = gtsam.Point3(points[pt].cartesian[0],points[pt].cartesian[1],points[pt].cartesian[2])#xyz
    #         factor_graph.push_back(gtsam.PriorFactorPoint3(
    #             P(points[pt].id),
    #             cartesian_point,
    #             gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1]))#gtsam.noiseModel.Gaussian.Covariance(np.array(points[pt].covariance))# put more uncertainty in altitude...
    #         ))
    #         factor_count += 1
    #         values.insert(P(pt), cartesian_point)

    #     srt_rm = factor_count
    #     # ----------- Detected keypoints: -----------
    #     for pt in range(len(points)):
    #         for observation in points[pt].observations:
    #             camera_factor_idx = (frame_idx_map.index(observation.frame_idx))*self.data_manager.config.nrcams + observation.camera_idx
    #             uv = gtsam.Point2(observation.uv[0],observation.uv[1])
    #             noise = gtsam.noiseModel.Isotropic.Sigmas((np.array([5,5])))
    #             model = gtsam.noiseModel.Robust.Create(
    #                 gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    #                 noise
    #             )
    #             factor_graph.push_back(gtsam.GeneralSFMFactor2Cal3Fisheye(
    #                 uv,
    #                 model,
    #                 C(camera_factor_idx),
    #                 P(points[pt].id),# == P(pt)
    #                 K(observation.camera_idx)
    #             ))
    #             factor_count += 1
    #             if frame_idx_map.index(observation.frame_idx) == 0:
    #                 prev_factors.append(factor_count)
        


    #     # ========================= Optimization =========================
    #     params = gtsam.LevenbergMarquardtParams()
    #     params.setVerbosityLM("ERROR")
    #     lm = gtsam.LevenbergMarquardtOptimizer(factor_graph, values, params)
    #     result = lm.optimize()
    #     marginals = gtsam.Marginals(factor_graph, result)

    #     t1 = time.time()
    #     print("InitialOptimization: {:.1f} ms".format((t1-t0)*1000))


    #     # ========================= Update Values =========================
    #     # Pose:
    #     for i in range(len(frames)):
    #         frames[i].T_origin2body_estimated = self.gtsam2pose(result.atPose3(X(i)))

    #     # MapPoints
    #     for pt in range(len(points)):
    #         opt_pt = result.atPoint3(P(points[pt].id))

    #         data['map'].mappoints[pt].cartesian = opt_pt

    #     # NOTE in initial optimization, we do not update calibration
    #     print("    Setting pose index to {}".format(pose_idx))
        
    #     data['map'].factor_graph = factor_graph
    #     data['map'].values = values
    #     data['map'].pose_idx = pose_idx

    #     # isam_params = gtsam.ISAM2Params()
    #     # isam_params.setFactorization("CHOLESKY")
    #     # isam_params.relinearizeSkip = 10
    #     # self.isam_optimizer = gtsam.ISAM2(isam_params)
    #     # # print(factor_graph)

    #     # self.isam_optimizer.update(factor_graph, values)
    





    ################################################################################################################
    #                                    Methods for adding to factor graph
    ################################################################################################################


    def add_pose(self, factor_graph, values, pose_idx, pose_xn, noise, custom_between=False):
        # pose_xn = self.pose2gtsam(currentMultiframe.T_origin2body_estimated)
        factor_graph.addPriorPose3(X(pose_idx), pose_xn, noise)
        if values is not None:
            values.insert(X(pose_idx), pose_xn)

        # Add imu2cam extrinsic between factors
        for cam in range(self.data_manager.config.nrcams):
            camera: PinholeCamera = self.data_manager.config.cameras[cam]
            cam_idx = pose_idx*self.data_manager.config.nrcams + cam
            prior_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0,0,0,0,0,0]))

            # IMU to camera prior using custom factor:
            if custom_between:
                factor_graph.add(gtsam.CustomFactor(prior_between_noise, [X(pose_idx), C(cam_idx), Y(cam)],partial(error_cal_between, [None])))
            else:
                factor_graph.add(gtsam.BetweenFactorPose3(X(pose_idx),C(cam_idx),self.pose2gtsam(camera.base2cam),prior_between_noise))

            if values is not None:
                values.insert(C(cam_idx), pose_xn.compose(self.pose2gtsam(camera.base2cam)))
    
    def add_cameras(self, factor_graph, values, intrinsic_noise, extrinsic_noise):
        for cam in range(self.data_manager.config.nrcams):
            # Intrinsics:
            camera: PinholeCamera = self.data_manager.config.cameras[cam]
            init_cal = gtsam.Cal3Fisheye(camera.K_[0][0], camera.K_[1][1],0,camera.K_[0][2],camera.K_[1][2],camera.D4[0],camera.D4[1],camera.D4[2],camera.D4[3])
            factor_graph.push_back(gtsam.PriorFactorCal3Fisheye(
                K(cam),
                init_cal,
                intrinsic_noise
            ))
            if values is not None:
                values.insert(K(cam), init_cal)
        
            # Extrinsics:
            extrinsic = self.pose2gtsam(camera.base2cam)
            factor_graph.addPriorPose3(Y(cam), extrinsic, extrinsic_noise)
            if values is not None:
                values.insert(Y(cam), extrinsic)
    
    def add_imu(self, factor_graph, values, pose_idx, summed_imu_z):
        # Create IMU factor
        factor_graph.push_back(gtsam.ImuFactor(
            X(pose_idx-1), V(pose_idx-1),
            X(pose_idx), V(pose_idx),
            B(pose_idx-1),
            summed_imu_z
        ))

        # Bias between: (constant)
        factor_graph.push_back(gtsam.BetweenFactorConstantBias(
            B(pose_idx-1),
            B(pose_idx),
            gtsam.imuBias.ConstantBias(),
            self.sigma_b
        ))
        

    def add_point(self, factor_graph, values):
        pass




    def pose2gtsam(self,pose: Pose):
        q = pose.q
        gtsam_R = gtsam.Rot3(q[0],q[1],q[2],q[3]) # w,x,y,z
        gtsam_pose = gtsam.Pose3(gtsam_R, gtsam.Point3(pose.t[0][0], pose.t[1][0], pose.t[2][0]))
        return gtsam_pose
    
    def state2gtsam(self,x):
        q = euler.euler2quat(x[3], x[4], x[5])
        gtsam_R = gtsam.Rot3(q[0],q[1],q[2],q[3]) # w,x,y,z
        gtsam_pose = gtsam.Pose3(gtsam_R, gtsam.Point3(x[0], x[1], x[2]))
        return gtsam_pose

    def gtsam2pose(self,gtsam_pose):
        t = gtsam_pose.translation()
        R = gtsam_pose.rotation().matrix()
        return Pose(t,R)
    