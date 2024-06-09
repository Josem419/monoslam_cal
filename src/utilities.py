#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu


from re import match
import yaml
import gtsam
from threading import Thread
import multiprocessing
import math

from data_manager import DataManager, SLAMConfig
from multiframe import MultiFrame
from pose import *

from itertools import compress

from typing import List, Optional, Union

# give it access to the global singleton to make things easy
data_manager = DataManager()

def associate_by_triangulation(matches, cam: int, queryFrame: MultiFrame, trainFrame: MultiFrame, settings):
    """
    Given matches between frames, confirm those matches via triangulation given the estimated pose
    Also return the triangulated points in the queryFrame
    queryFrame is meant to be the reference frame / more trusted frame
    """
    
    T_origin2cam_init = queryFrame.T_origin2body_estimated.convert(data_manager.config.cameras[cam].base2cam)

    # get features:
    uv_init = np.array([queryFrame.feature_pts[cam][match.queryIdx] for match in matches])
    uv_curr = np.array([trainFrame.feature_pts[cam][match.trainIdx] for match in matches])
    # og_indexing = np.arange(len(uv_init))

    # ------------ Un-project in first frame: ------------
    uv_vector = data_manager.config.cameras[cam].image2vector(uv_init)
    uv_vector_world = (T_origin2cam_init.R@uv_vector.T).T # in world frame
    
    # Resolve scale by assuming ground at 0 altitude
    # s = ((0-T_origin2cam_init.t[2][0]) / uv_vector_world[:,2]).reshape(-1,1)

    xyzPoint = uv_vector_world + T_origin2cam_init.t[:,0] # t_world2point = t_cam2point - t_cam2world

    # ------------ Test for reprojection error in 2nd frame: ------------
    uv_curr_reproj, valid = data_manager.config.cameras[cam].projectPoints(xyzPoint,trainFrame.T_origin2body_estimated)

    reproj_err_sq = np.linalg.norm(uv_curr_reproj-uv_curr, axis=1)
    correct = reproj_err_sq < settings['min_reproj_err']**2 #TODO define based on resolution and gt pose accuracy
    
    # ...and filter for those below threshold:TODO delete ones that are not needed
    # og_indexing = og_indexing[correct]
    correct = np.all([correct, valid], axis=0) # filter for those below threshold and for badly projected points:
    uv_init = uv_init[correct]
    uv_curr = uv_curr[correct]
    xyzPoint = xyzPoint[correct]
    matches = list(compress(matches, correct))


    return matches, uv_init, uv_curr, xyzPoint
