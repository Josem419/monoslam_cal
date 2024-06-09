#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu

from ctypes.wintypes import POINT
from cupshelpers import Printer
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

import cv2
import time
from pose import *
from typing import Optional, Union, List
from scipy.optimize import least_squares


class Camera(ABC):
    "Class for a pinhole camera model"

    def __init__(self, frame_id: str, w: int, h: int, base2cam: np.ndarray) -> None:

        self.width = w
        self.height = h
        self.frame_id: str = frame_id

        # is a translation and rotation rigid transformation matrix from a base link to the camera
        # could be an identity to signify that the camera is the baselink
        self.base2cam: Pose = base2cam

    @abstractmethod
    def image2vector(self):
        pass


class View:
    def __init__(self, camera: Camera, pose: Pose, img) -> None:
        self.camera: Camera = camera
        self.pose: Pose = pose
        self.image: np.ndarray = img


class PinholeCamera(Camera):

    def __init__(
        self,
        frame_id: str,
        w: int,
        h: int,
        K: np.ndarray,
        D: np.ndarray,
        base2cam: Pose,
    ) -> None:
        super().__init__(frame_id, w, h, base2cam)

        D_temp = D.squeeze()

        # validate the K or D input data
        if K.shape != (3, 3) or len(D_temp) < 4:
            raise Exception(
                "K or D matrices not provided in right format. Expecting K:3x3 and D: 1x4,5,8"
            )

        self.K_: np.ndarray = K
        self.D_: np.ndarray = D_temp

        # run regression over points to go from 1xn to 1x4 distortion vector
        if len(self.D_) > 4:
            # Run regression over the points:
            missing_coeff = 8 - len(np.squeeze(self.D_))
            distortion_matrix = np.concatenate(
                (np.squeeze(self.D_), np.zeros(missing_coeff))
            )
            rs = np.array([*range(0, w, 10)])

            def proj_dist_full(r):
                r2 = r**2
                r_dist = (
                    1
                    + distortion_matrix[0] * r2
                    + distortion_matrix[1] * r2**2
                    + distortion_matrix[4] * r2**3
                ) / (
                    1
                    + distortion_matrix[5] * r2
                    + distortion_matrix[6] * r2**2
                    + distortion_matrix[7] * r2**3
                )
                return r_dist

            def proj_dist_4(r, params):
                r2 = r**2
                r_dist = 1 + params[0] * r2 + params[1] * r2**2
                return r_dist

            def loss(params):
                res = []
                for r in rs:
                    res.append(proj_dist_full(r) - proj_dist_4(r, params))
                return res

            opt = least_squares(loss, [0, 0], verbose=0, gtol=1e-3)

            if opt.success:
                params_s = opt.x
                self.D4 = np.array([params_s[0], params_s[1], self.D_[2], self.D_[3]])
            else:
                raise Exception(
                    "Perspective Camera: optimization for 1x4 distortion array failed"
                )

        elif len(self.D_) == 4:
            # make a 4 element vector for distortion
            self.D4 = np.squeeze(self.D_)
        else:
            raise Exception("D matrix must have >= 4 values")

        # set the

    def projectPoints(
        self,
        objectPoints: Union[List, np.ndarray],
        T_origin2base: Pose,
        T_origin2cam: Optional[Pose] = None,
        padding: int = 0,
    ):
        """
        Projects points into the image, and returns image points and if they are valid (in image) (using OpenCV)
        :param objectPoints are a 2D list/np array of 3D cartesian points to project into the image
        :param T_origin2base: Can either specify only the base pose T_origin2base (in which case it will transform it to the camera pose using the extrinsics)
        :param T_origin2cam: Or you can set the camera pose T_origin2cam, in which case T_origin2base will be ignored
        :param padding: will return reprojected points outside image by this value
        TODO optionally return covariance (opencv returns jacobian)
        """
        if objectPoints is None:
            return [], []
        else:
            objectPoints = np.array(objectPoints)
        if len(objectPoints) == 0:
            return [], []

        # elif len(objectPoints) == 1:
        #     objectPoints

        # Have the option to supply the camera pose directly:
        if T_origin2cam is not None:
            R_cam2origin, t_cam2origin = T_origin2cam.invert().Rt()
        else:
            R_cam2origin, t_cam2origin = (
                T_origin2base.convert(self.base2cam).invert().Rt()
            )

        """
        # NON OPENCV METHOD:
        missing_coeff = 8-len(np.squeeze(self.D_))
        distortion_matrix = np.concatenate((np.squeeze(self.D_), np.zeros(missing_coeff)))

        # Transform point to camera frame:
        objpts_cam = (R_cam2origin@(objectPoints-t_origin2cam).T).T

        # Distort and project:
        x_ = objpts_cam[:,0]/objpts_cam[:,2]
        y_ = objpts_cam[:,1]/objpts_cam[:,2]
        r2 = x_**2 + y_**2
        r_dist = (1 + distortion_matrix[0]*r2 + distortion_matrix[1]*r2**2 + distortion_matrix[4]*r2**3) / (1 + distortion_matrix[5]*r2 + distortion_matrix[6]*r2**2 + distortion_matrix[7]*r2**3)
        
        x__ = x_*r_dist + 2*distortion_matrix[2]*x_*y_ + distortion_matrix[3]*(r2 + 2*x_**2)
        y__ = y_*r_dist + 2*distortion_matrix[3]*x_*y_ + distortion_matrix[2]*(r2 + 2*y_**2)

        xyz__=np.ones([len(y__),3])
        xyz__[:,0]=x__
        xyz__[:,1]=y__

        reproj_obj_pts = ((self.K@xyz__.T).T)[:,:2]
        """

        cam2origin_transform = np.hstack((R_cam2origin, t_cam2origin))
        cam2origin_transform = np.vstack((cam2origin_transform,np.array([0,0,0,1])))
        # make object points homogenous
        objectPoints = np.hstack((objectPoints, np.ones((objectPoints.shape[0], 1))))

        objpts_cam = (cam2origin_transform @ objectPoints.T).T

        # using opencv to not deal with distortion myself
        reproj_obj_pts, _ = cv2.projectPoints(
            objectPoints[:,:3], R_cam2origin, t_cam2origin, self.K_, self.D_
        )
        reproj_obj_pts = reproj_obj_pts[:, 0, :]

        # Check for projection  of points behind the image:
        # valid points are only those with a positive z value in the camera frame (ie in front of the camera)
        valid_proj = objpts_cam[:, 2] > 0
        valid_img = np.all(
            abs(reproj_obj_pts - np.array([self.width / 2, self.height / 2]))
            < np.array([self.width / 2 + padding, self.height / 2 + padding]),
            axis=1,
        )
        valid = np.all([valid_proj, valid_img], axis=0)

        return reproj_obj_pts, valid

    def image2vector(self, imagePoints):
        """
        Will project the imaged points into 3D cartesian vectors in the camera frame (scale is ambiguous)
        :param imagePoints must be a 2D array!
        """

        # first undistort:
        # NOTE: consider using fisheye cv2 class, or custom optimized function. This can be slightly inaccurate...
        if imagePoints.shape[0] == 1:
            imagePoints_rect = cv2.undistortPoints(
                imagePoints, self.K_, self.D_, P=self.K_
            )[0]
        else:
            imagePoints_rect = np.squeeze(
                cv2.undistortPoints(imagePoints, self.K_, self.D_, P=self.K_)
            )

        # imagePoints_rect0 = np.concatenate((imagePoints_rect,np.ones((imagePoints_rect.shape[0],1))), axis=1)
        imagePoints_rect_full = np.ones([imagePoints_rect.shape[0], 3])
        imagePoints_rect_full[:, :2] = imagePoints_rect

        XYZ = (np.linalg.inv(self.K_) @ imagePoints_rect_full.T).T
        norm = np.linalg.norm(XYZ, axis=1)
        XYZ = XYZ / norm[:, None]
        return XYZ

    def plane_intersection(self):
        pass
