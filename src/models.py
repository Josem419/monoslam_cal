#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu

import numpy as np


class FundamentalMatrixTransform(object):
    """
    Need a model to fit a fundamental matrix
    """

    def __init__(self):
        # estimates a fundamental matrixusing RANSAC and the normalized 8 point algo
        self.params = np.eye(3)

    def __call__(self, coords):
        # needed for ransac
        coords_homogeneous = np.hstack((coords, np.ones(coords.shape[0])))
        return coords_homogeneous @ self.params.T

    def lls_eight_point_alg(self, points1, points2):
        """
        LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using
        linear least squares eight point algorithm
        Arguments:
            points1 - N points in the first image that match with points2
            points2 - N points in the second image that match with points1

            Both points1 and points2 are from the get_data_from_txt_file() method
        Returns:
            F - the fundamental matrix such that (points2)^T * F * points1 = 0
        Please see lecture notes and slides to see how the linear least squares eight
        point algorithm works
        """

        # w = np.ones((points1.shape[0], 9))
        # w[:, :2] = points1
        # w[:, :3] *= points2[:, 0, np.newaxis]
        # w[:, 3:5] = points1
        # w[:, 3:6] *= points2[:, 1, np.newaxis]
        # w[:, 6:8] = points1

        # combine the points into a single W vector
        w = []

        for idx, pt in enumerate(points1):
            # make a row vector for the W entry
            row = []

            for v in pt:
                for v_p in points2[idx]:
                    row.append(v * v_p)

            w.append(row)

        w = np.array(w)

        # SVD of W to get the  null space of w
        u, s, vt = np.linalg.svd(w)
        f_hat = vt[-1]  # null space of w

        # SVD again to force Fhat to a rank2 matrix
        up, sp, vtp = np.linalg.svd(np.reshape(f_hat, (3, 3)).T)

        # get rid the smallest value in the sigular values and multiply togeher to get F
        sp[-1] = 0
        F = np.matmul(up, np.matmul(np.diag(sp), vtp))

        return F

    ### dont rename, needed for ransac
    def estimate(self, points1, points2):

        # Setup homogeneous linear equation as point1.T * F * point2 = 0.

        # NORMALIZE (for well conditioned matrices) 8 point algo
        # make the transformation matrices that I will fill in the translation and scaling
        t = np.eye(3)
        tp = np.eye(3)

        # first get centroid of each points and add the translation
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)

        t[:, -1] = -c1.T
        tp[:, -1] = -c2.T
        t[-1, -1] = 1
        tp[-1, -1] = 1

        # get the scaling factor and add it to the transform
        s1 = points1[:, :2]  # drop the last term
        s1 = s1 - c1[:2]  # subtract element wise by the mean for x and y
        s1 = np.linalg.norm(s1, axis=1) ** 2
        s1 = np.sum(s1)
        s1 = np.sqrt((2 * points1.shape[0]) / s1)

        s2 = points2[:, :2]  # drop the last term
        s2 = s2 - c2[:2]  # subtract element wise by the mean for x and y
        s2 = np.linalg.norm(s2, axis=1) ** 2
        s2 = np.sum(s2)
        s2 = np.sqrt((2 * points2.shape[0]) / s2)

        # add the scale factor to the transformation matrix. Scale the translation as well
        t[0, 0] = s1
        t[1, 1] = s1
        t[0, 2] = s1 * t[0, 2]
        t[1, 2] = s1 * t[1, 2]

        tp[0, 0] = s2
        tp[1, 1] = s2
        tp[0, 2] = s2 * tp[0, 2]
        tp[1, 2] = s2 * tp[1, 2]

        # transform the points
        norm_pts1 = np.matmul(points1, t.T)
        norm_pts2 = np.matmul(points2, tp.T)

        # get the new fundamental matrix using the normalized points
        F = self.lls_eight_point_alg(norm_pts1, norm_pts2)

        # denormalize the returned F matrix
        F = tp.T @ F @ t

        self.params = F

        return True

    # dont rename needed for ransac
    def residuals(self, points1, points2):
        # # Compute the Sampson distance.
        # points1_homogeneous = np.column_stack([points1, np.ones(points1.shape[0])])
        # points2_homogeneous = np.column_stack([points2, np.ones(points2.shape[0])])

        F_points1 = self.params @ points1.T
        Ft_points2 = self.params.T @ points2.T

        points2_F_points1 = np.sum(points2 * F_points1.T, axis=1)

        return np.abs(points2_F_points1) / np.sqrt(
            F_points1[0] ** 2
            + F_points1[1] ** 2
            + Ft_points2[0] ** 2
            + Ft_points2[1] ** 2
        )

