#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu

from scipy.linalg import block_diag
from typing import Optional, List
import numpy as np
import cv2

import transforms3d.quaternions as quaternions
import transforms3d.euler as euler


class Pose:

    def __init__(
        self,
        t: np.ndarray = None,
        rotation: np.ndarray = None,
        covariance: np.ndarray = None,
    ) -> None:

        # Initialize translation vector
        self.t: np.ndarray = np.zeros(3, dtype=np.float64).reshape(3, 1)

        # Initialize rotation matrix
        self.R: np.ndarray = np.zeros((3, 3), dtype=np.float64)

        # Alternate rotation representations
        # TODO consider making these methods instead
        self.q: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion rotation
        self.rvec: np.ndarray = np.zeros(3, dtype=np.float64)  # rodrigues rotation
        self.euler_extrinsic: np.ndarray = np.zeros(
            3, dtype=np.float64
        )  # euler extrinsic (sxyz) angles
        self.covariance: Optional[np.ndarray] = (
            None  # Pose covariance (3 translational, 3 rotational euler angles)
        )

        # update with the passed in values
        if t is not None and rotation is not None:
            self.update(t, rotation, covariance)

    def update(
        self, t: np.ndarray, rotation: np.ndarray, covariance: np.ndarray = None
    ):
        """
        Update dataclass with pose params. Note all inputs have to be np arrays
        :param rotation: either rotation matrix (3,3), quaterions (4) or euler (3) extrinsic angles (will automatically detect)
        :param t: translation vector (3,1)
        :param covariance: optional to be specified. (6,6) or (36) (translational + angular covariance
        """
        self.t = t.reshape(3, 1)
        if rotation.shape == (3, 3):
            self.R = rotation
            self.q = quaternions.mat2quat(rotation)
            self.euler_extrinsic = euler.mat2euler(rotation)
            self.rvec = mat2rvec(self.R)
        elif rotation.shape == (4,):
            self.q = rotation
            self.R = quaternions.quat2mat(rotation)
            self.euler_extrinsic = euler.mat2euler(self.R)
            self.rvec = mat2rvec(self.R)
        elif rotation.shape == (3,):
            self.euler_extrinsic = rotation
            self.R = euler.euler2mat(rotation[0], rotation[1], rotation[2])
            self.q = euler.euler2quat(rotation[0], rotation[1], rotation[2])
            self.rvec = mat2rvec(self.R)
        else:
            print(
                "error, wrong rotation shape: {}. Should be one of (3,3) (4,) (3,)".format(
                    rotation.shape
                )
            )

        if covariance is not None:
            if covariance.shape == (6, 6) or covariance.shape == 36:
                self.covariance = covariance.reshape(6, 6)
            else:
                print(
                    "error, Covariance shape is unexpected: {}".format(covariance.shape)
                )

    def Rt(self):
        """
        Return the rotation matrix and translation vector
        """
        return self.R, self.t

    def transposedCovariance(self, covariance=None):
        """
        returns the covariance as 6x6 rotation,translation, instead of this internal 6x6 translation,rotation representation:
        This is necessary because GTSAM takes rotation and then translation in the covariance
        """
        if covariance is None:
            covariance = self.covariance
        transposed_covariance = np.zeros((6, 6))
        transposed_covariance[:3, :3] = covariance[3:, 3:]
        transposed_covariance[3:, 3:] = covariance[:3, :3]
        transposed_covariance[3:, :3] = covariance[:3, 3:]
        transposed_covariance[:3, 3:] = covariance[3:, :3]

        return transposed_covariance

    def setFromTransposedCovariance(self, covariance):
        self.covariance = self.transposedCovariance(covariance=covariance)

    def invert(self):
        """
        Return an inverted pose object
        """

        # Putting pose in a format to also change covariance:
        pose = [
            self.t[0][0],
            self.t[1][0],
            self.t[2][0],
            self.euler_extrinsic[0],
            self.euler_extrinsic[1],
            self.euler_extrinsic[2],
        ]

        # Covariance conversion:
        cov_i = None
        if self.covariance is not None:
            
            # defining conversion function TODO define analytical jacobian instead
            def _invert(pose, format=True):
                R = euler.euler2mat(pose[3], pose[4], pose[5])
                t = np.array([pose[:3]]).T
                R_i, t_i = InvertRt(R, t)

                if format:
                    r, p, y = euler.mat2euler(R_i)
                    return t_i[0][0], t_i[1][0], t_i[2][0], r, p, y
                else:
                    return t_i, R_i
                
            # compute the jacobian
            J_inv = numJac3Pts(_invert, pose)
            cov_inv = J_inv @ self.covariance @ J_inv.T

        # invert the transform itself
        R_inv, t_inv = InvertRt(self.R, self.t)

        transform_inv = Pose()
        transform_inv.update(t_inv, R_inv, cov_i)

        return transform_inv

    def convert(self, T_child2target):
        """
        Converts a pose:    T_source2child  (source frame)
        to pose:            T_source2target (source frame)
        by defining:        T_child2target  (child frame)

        :param T_child2target: Pose object defining the child pose to target pose IN CHILD FRAME
        :return T_child2target: new Pose object defining the source to target frame IN SOURCE FRAME
        """
        # Putting pose in a format to change covariance:
        euler_source2child = self.euler_extrinsic
        t_source2child = self.t
        # [t,r, t,r]
        params = [
            t_source2child[0][0],
            t_source2child[1][0],
            t_source2child[2][0],
            euler_source2child[0],
            euler_source2child[1],
            euler_source2child[2],
            T_child2target.t[0][0],
            T_child2target.t[1][0],
            T_child2target.t[2][0],
            T_child2target.euler_extrinsic[0],
            T_child2target.euler_extrinsic[1],
            T_child2target.euler_extrinsic[2],
        ]

        # defining conversion function TODO define analytical jacobian instead
        def child2target(params, format=True):
            # Unpacking:
            R_source2child = euler.euler2mat(params[3], params[4], params[5])
            t_source2child = np.array([params[:3]]).T
            R_child2target = euler.euler2mat(params[9], params[10], params[11])
            t_child2target = np.array([params[6:9]]).T

            # Pose transform:
            R_source2target = R_source2child @ R_child2target
            t_source2target = t_source2child + R_source2child @ t_child2target

            if format:
                r, p, y = euler.mat2euler(R_source2target)
                return (
                    t_source2target[0][0],
                    t_source2target[1][0],
                    t_source2target[2][0],
                    r,
                    p,
                    y,
                )
            else:
                return t_source2target, R_source2target

        # Covariance conversion:
        cov_source2target = None
        if self.covariance is not None:
            if T_child2target.covariance is None:
                cov = block_diag(*[self.covariance, np.zeros((6, 6))])
            else:
                cov = block_diag(*[self.covariance, T_child2target.covariance])
            J_child2target = numJac3Pts(child2target, params)
            cov_source2target = J_child2target @ cov @ J_child2target.T

        t_source2target, R_source2target = child2target(params, False)

        T_source2target = Pose(t_source2target, R_source2target, cov_source2target)

        return T_source2target

    def output(self) -> dict:
        output = {}
        output.update(tvec=self.t)
        output.update(euler=self.euler_extrinsic)

        return output


"""
Utility Functions
"""


def rvec2mat(rvec: np.ndarray):
    """
    Converts a Rodrigues rotation vector to a 3x3 matrix
    :param rvec: 3d rotation vector
    :return: corresponding 3x3 rotation matrix
    """
    return cv2.Rodrigues(rvec)[0]


def mat2rvec(R: np.ndarray):
    """
    Converts a matrix to a Rodrigues rotation vector
    :param rot: a 3x3 rotation matrix
    :return: the 3D rotation vector
    """
    return cv2.Rodrigues(R)[0].reshape(3)


def InvertRt(R, t):
    """
    Invert the frame transform (rotation matrix and translation vector):
    R_ = R.T   (R.inv() = R.T rotation matrix is orthonormal ))
    t_ = R_@-t
    """
    R_ = R.T
    t_ = R_ @ -t
    return R_, t_


# reutn a jacobian for 3 points
def numJac3Pts(f, x, kwargs={}, delta=1e-6, x_indices=None, anglularIndices=None):
    x = np.array(x)
    wrapped_f = lambda x: np.array(f(x, **kwargs)).ravel()

    n = x.shape[0]
    dx = np.diag(np.full(n, delta))

    if x_indices is not None:
        # Restricting the gradient computation to the specified indices of the input vector x
        dx = dx[x_indices]

    fwd = np.vstack(map(wrapped_f, x + dx))
    bwd = np.vstack(map(wrapped_f, x - dx))

    res = fwd - bwd
    if anglularIndices is not None:
        if x_indices is not None:
            anglularIndices = set(anglularIndices) - set(x_indices)

        anglularIndices = np.array(anglularIndices)
        res[:, anglularIndices] = angle_diff(
            fwd[:, anglularIndices], bwd[:, anglularIndices]
        )

    res /= 2 * delta
    res = res.T

    return res


def angle_diff(angles, ref_angles):
    """
    Computes the difference between a set of angular values and a reference value
    :param angles: angle values in the form of an array or a single scalar value
    :param ref_angle: the reference angle from which the differnces are computed
    :return: the angular difference between angles and ref angle. The matches the type of angles (scalar or array)
    """
    angles = np.array(angles)
    ref_angles = np.array(ref_angles)
    angles_is_number = angles.shape == ()
    ref_angles_is_number = ref_angles.shape == ()
    unpack = angles_is_number and ref_angles_is_number
    if angles.shape != ref_angles.shape:
        if angles_is_number:
            angles = np.full_like(ref_angles, angles)
        elif ref_angles_is_number:
            ref_angles = np.full_like(angles, ref_angles)
        else:
            raise ValueError("shape mismatch between angle and ref_angle arrays")

    res = angles - ref_angles

    while True:
        mask = res <= -np.pi
        if np.any(mask):
            res[mask] += 2 * np.pi
        else:
            break

    while True:
        mask = res > np.pi
        if np.any(mask):
            res[mask] += -2 * np.pi
        else:
            break

    # if a scalar return only the first value
    return res[0] if unpack else res
