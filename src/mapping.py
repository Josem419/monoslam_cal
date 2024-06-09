#!/usr/bin/python3

# Jose Medina
# josem419@stanford.edu

import numpy as np
import gtsam
from threading import Thread, Lock
from typing import List, Optional

import time
import cv2
from itertools import compress
import copy

from data_manager import DataManager
from multiframe import MultiFrame
from utilities import associate_by_triangulation
from optimizer import Optimizer

# TODO create class/struct for observations?
class Observation:
    def __init__(
        self, camera_idx, frame_idx, uv: np.ndarray, descriptor, score
    ) -> None:
        self.camera_idx = camera_idx
        self.frame_idx = frame_idx
        self.uv = uv
        self.descriptor = descriptor
        self.score = score


class MapPoint:

    def __init__(self, cartesian_pt: np.ndarray, covariance: np.ndarray) -> None:

        self.id: Optional[int] = None
        self.cartesian: np.ndarray = cartesian_pt
        self.covariance: np.ndarray = covariance
        self.observations: List[Observation] = []
        self.most_recent_frame: int = None

    def addObservation(
        self,
        camera_idx: int,
        frame_idx: int,
        uv: np.ndarray,
        descriptor,
        score: float = 0,
    ):

        self.observations.append(
            Observation(camera_idx, frame_idx, uv, descriptor, score)
        )
        if self.most_recent_frame is None:
            self.most_recent_frame = frame_idx
        else:
            self.most_recent_frame = max(self.most_recent_frame, frame_idx)


class Map:

    def __init__(self) -> None:

        self.mutex = Lock()
        self.mappoints: List[MapPoint] = []
        self.point_counter = 0
        self.id = 0

        # Mapping GTSAM optimization variables:
        self.factor_graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.pose_idx = 0

    def addMapPoint(self, mappoint: MapPoint):

        # Assign id to keep track:
        mappoint.id = self.point_counter
        self.point_counter += 1

        self.mappoints.append(mappoint)

    def addMapPoints(self, mappoints: List[MapPoint]):
        # NOTE this assumes the index has been set already
        self.point_counter += len(mappoints)
        self.mappoints.extend(mappoints)




class LocalMapper:

    def __init__(self) -> None:
        self.data_manager = DataManager()

        self.settings = {"mapping_seach_n_frames":1,
                         "mapping_min_desc_distance":300,
                         "max_reproj_dist":10000,
                         "min_reproj_err":20}
        
        self.tentative_new_mappoints = []
        self.tentative_new_frames = []

        self.optimizer = Optimizer()



    def run(self):
        print("                                         Starting mapping thead")

        
        while not self.data_manager.shutdown:
            
            # Resetting: Freezes tracking until it is reset here...
            if self.data_manager.config.reset:
                self.data_manager.reset_data()
                self.data_manager.config.index = 0 # TODO do not reset this, but another variable ...? Need to keep track of imus - frame correspondencies
                self.data_manager.config.reset = False
                time.sleep(0.05)
                continue
            
            # If tracking has initialized and map is active:
            if self.data_manager.data.currentMap is not None:
                print("                                         map present")
                

                if not self.process_MKFs():
                    time.sleep(0.05)
                    continue # no frames to process
                print("                                         {} frames processed".format(len(self.tentative_new_frames)))

                # map point culling

                # MKF culling

                data = {}
                data['new_frames'] = self.tentative_new_frames
                if (len(self.data_manager.data.MKFs) == 0):
                    data['new_mappoints'] = self.data_manager.data.currentMap.mappoints # initial points mapped in tracker
                else:
                    data['new_mappoints'] = self.tentative_new_mappoints
                self.optimizer.BundleAdjustment(data)
                self.data_manager.data.MKFs.extend(self.tentative_new_frames)

                # Loop closing (within this map)

            else:
                print("                                         map None")




            self.frames_idxs_to_map = []
            time.sleep(0.2)# TODO

    def process_MKFs(self):
        """
        Loops through process_queue, computes BoW, inserts into MKFs
        To limit size of process_queue, 
        Do accept keyframes:
            NOT loop closure/local mapping is frozen
            more than x frames have passed since relocalization
            mapping is idle (interrpts BA, but does not insert yet)
                
                more than x frames since last keyframe (ORBSLAM is every 0.66 seconds)
                OR more than x frames since
                AND <90% of points tracked in ref
        """
        if self.data_manager.data.process_queue.empty():
            return False
        print("                                         Starting mapping =================")
        
        self.data_manager.data.local_mapping_proc = True # cannot add new keyframes
        self.tentative_new_mappoints = []
        self.tentative_new_frames = []
        while not self.data_manager.data.process_queue.empty():
            multiframe: MultiFrame = self.data_manager.data.process_queue.get_nowait()

            #TODO BoW processing

        # Map new points:
        if len(self.data_manager.data.MKFs) > 7: # TODO
            self.mapNewPoints(multiframe)

        self.tentative_new_frames.append(multiframe)
        
        # print("Mapping: {:.1f} ms".format((t1-t0)*1000))
        
        self.data_manager.data.local_mapping_proc = False # allow new keyframes
        return True
    
    def mapNewPoints(self, this_MKF: MultiFrame):

        """
            
        Given a current MKF:
        orb feature matching to previus frames
        for each match:
            if match.query and match.train are not associated:
                triangulate
        
        # TODO:
        Fuse between cameras
        fuse with existing map points if necessary

        """

        matcher = cv2.BFMatcher()

        # Look in nearest n frames:
        pt_idx = self.data_manager.data.currentMap.point_counter
        for i in range( len(self.data_manager.data.MKFs)-1, max(  len(self.data_manager.data.MKFs)-(2+self.settings['mapping_seach_n_frames']), -1  ), -1): # propagate backwards:

            other_MKF = self.data_manager.data.MKFs[i]

            for cam in range(self.data_manager.config.nrcams):
                # Standard ORB matching:
                #                       query                      train
                matches = matcher.match(other_MKF.descriptors[cam],this_MKF.descriptors[cam])
                matches = sorted(matches, key = lambda x:x.distance)
                
                for i in range(len(matches)):
                    if matches[i].distance > self.settings['mapping_min_desc_distance']:
                        break
                del matches[i:]

                # match.queryIdx, match.trainIdx

                # Sort for only features that were un-associated in both images:
                new_matches = [match for match in matches if not this_MKF.associations[cam][match.trainIdx] and not other_MKF.associations[cam][match.queryIdx]]

                new_matches, uv_init, uv_curr, xyzPoint = associate_by_triangulation(new_matches,cam,other_MKF,this_MKF,{'max_reproj_dist':self.settings['max_reproj_dist'], 'min_reproj_err':self.settings['min_reproj_err']})


                # ======================= Fuse map points, create map points and add to Map: =======================
                # first filter for mappoints that have been observed in last n frames
                # TODO include to fuse with map points in self.tentative_new_mappoints for every different camera & frame
                n = 50
                mappoint_indices = np.array([i for i in range(len(self.data_manager.data.currentMap.mappoints)) if self.data_manager.data.currentMap.mappoints[i].most_recent_frame >= this_MKF.frame_idx-n])
                relevant_map_points = np.array([self.data_manager.data.currentMap.mappoints[i].cartesian for i in mappoint_indices])

                # Filter for mappoints that are visible in this frame using estimated pose:
                relevant_uvs, valid = self.data_manager.config.cameras[cam].cartesian2image(relevant_map_points,this_MKF.T_origin2body_estimated, padding=40)# use padding for uncertainty in pose
                inview_relevant_uvs = relevant_uvs[valid]
                inview_relevant_map_points = relevant_map_points[valid]
                inview_mappoint_indices = mappoint_indices[valid]

                debug_img = copy.deepcopy(this_MKF.imgs[cam])

                for i in range(len(xyzPoint)):
                    
                    # Check for coincident map points:
                    if len(inview_mappoint_indices) > 0:

                        """
                        We want to have very distant constraints for deciding if a new map point exists (want to avoid creating duplicate map points) -> put high values for reprojection error and mappoint proximity
                        But for adding observations, want to be sure that that observation really belongs to the map point -> put more stringent constraints for that TODO
                        """

                        # filter for if observations are within 20 pixels of eachother:
                        # reproj xyz (uv_curr) <-> reproj mappoint in current frame
                        reproj_err = np.linalg.norm(inview_relevant_uvs - uv_curr[i],axis=1)
                        filter_reproj_err = reproj_err < 20

                        # Filter for if points are within 20m of eachother
                        filter_proximity = np.all(np.abs(inview_relevant_map_points - xyzPoint[i])<np.array([20,20,20]), axis=1)


                        fuse_candidates = filter_reproj_err | filter_proximity
                        candidate_mappoint_indices = inview_mappoint_indices[fuse_candidates]
                        candidate_descs = [self.data_manager.data.currentMap.mappoints[i].observations[-1].descriptor for i in candidate_mappoint_indices]

                        # Filter for if they have similar descriptors:
                        if len(candidate_mappoint_indices) > 0:
                            desc_dist = np.linalg.norm(candidate_descs-this_MKF.descriptors[cam][new_matches[i].trainIdx],axis=1)
                            index_min = min(range(len(desc_dist)), key=desc_dist.__getitem__)
                            
                            min_mappoint = self.data_manager.data.currentMap.mappoints[candidate_mappoint_indices[index_min]]
                            uvs,_ = self.data_manager.config.cameras[cam].cartesian2image([min_mappoint.cartesian,[0,0,0]],this_MKF.T_origin2body)
                            # cv2.circle(debug_img,(int(uvs[0][0]),int(uvs[0][1])),10, [255,255,0],thickness=5, lineType=8, shift=0)
                            # cv2.circle(debug_img,(int(uv_curr[i][0]),int(uv_curr[i][1])),10, [255,0,255],thickness=5, lineType=8, shift=0)

                            # TODO if we are sure about fusing (more contrained reproj error & desc distance) -> just add observations to existing mappoints
                            # TODO if descriptors are far apart, still consider creating new map points? Maybe we don't want to have points that close...
                        
                        else:
                            uvs,_ = self.data_manager.config.cameras[cam].cartesian2image([xyzPoint[i],[0,0,0]],this_MKF.T_origin2body)
                            cv2.circle(debug_img,(int(uvs[0][0]),int(uvs[0][1])),10, [0,255,255],thickness=5, lineType=8, shift=0)

                            covariance = np.diag([10000,10000,10000])# Assume high covariance 100m sigma, gtsam will figure out the covariances, no need to calculate jacobians here
                            ptMap = MapPoint(xyzPoint[i], covariance)
                            ptMap.id = pt_idx
                            pt_idx+=1
                            ptMap.addObservation(cam,other_MKF.frame_idx,uv_init[i], other_MKF.descriptors[cam][new_matches[i].queryIdx])
                            ptMap.addObservation(cam,this_MKF.frame_idx,uv_curr[i], this_MKF.descriptors[cam][new_matches[i].trainIdx])
                            self.tentative_new_mappoints.append(ptMap)
                

                
            

