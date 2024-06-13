import multiprocessing.shared_memory
import numpy as np
from typing import List, Union
import cv2
import time
from pose import Pose

from data_manager import DataManager
from typing import Optional
import concurrent.futures
import multiprocessing


class Feature:

    def __init__(self, keypoint, descriptor) -> None:
        self.keypoint = keypoint
        self.descriptor = descriptor
        self.associated = False


class MultiFrame:

    def __init__(
        self,
        timestamp: float,
        imgs: List[np.ndarray],
        frame_idx: int,
        pose: np.ndarray = np.eye(4),
    ) -> None:

        # start time
        t0 = time.time()

        # not currently used
        self.data_manager = DataManager()

        self.frame_idx: int = frame_idx
        self.timestamp: float = timestamp

        # support multiple frames at a timestmep but for monoslam should only get one at a time
        self.imgs: List[np.ndarray] = []

        # Features for each camera:
        # self.features: List[List[Feature]] = [] # contains all of the below information:
        # self.features_cv: List[List[cv2.KeyPoint]] = []
        self.feature_pts: np.ndarray = np.zeros(1)
        self.descriptors = []
        self.associations = []
        self.keypoints = []
        # Estimated pose of SLAM
        # self.T_origin2body_estimated: Optional[Pose] = None
        # self.vel_estimated: Optional[np.ndarray] = None
        # self.vel_estimated_covariance = None

        self.pose = pose  # homogenous 4x4 identity matrix

        # Estimated imu bias:
        self.imu_bias = None
        self.imu_bias_covariance = None

        # GTSAM factors related to this frame:
        # TODO: fix my GTSAM implementation to actually make use of this
        self.factor_idxs = []

        # Orb feature extraction:
        ############################### METHOD 1 - CV2 Orb extractor ###############################
        # create the orb feature object
        orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )

        # for moncular there should only be  1 image.
        for img in imgs:
            if img is not None:

                self.imgs.append(
                    img
                )  # append to the list of images since its getting processed

                t1 = time.time()  # start of orb  extraction
                keypoints, descriptors = orb.detectAndCompute(img, None)
                # self.features_cv.append(keypoints)
                self.descriptors.append(descriptors)
                self.keypoints.append(keypoints)
                # self.features.append([Feature(keypoints[i],descriptors[i]) for i in range(len(keypoints))])
                self.associations.append([False for _ in range(len(keypoints))])
                # img2 = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)

                # parse out the keypoints into a np array for each point
                self.feature_pts = np.array(
                    [(cv_point.pt[0], cv_point.pt[1]) for cv_point in keypoints]
                )
            else:
                self.imgs.append(None)
                # self.features_cv.append([])
                self.descriptors.append([])
                self.feature_pts.append([])
                self.keypoints.append([])
                # self.features.append([])
                self.associations.append([])

        self.feature_pts = np.array(self.feature_pts)
        print("Number of images processed:" + str(len(self.imgs)))
        print(
            "[MKF] Frame Index: {} Time for feature extraction:  {} seconds".format(
                frame_idx, time.time() - t0
            )
        )

        # print("feature extraction: {:.1f} + {:.1f} ms".format((time.time()-t01)*1000.0, (t01-t0)*1000.0))

        ############################### METHOD 2  - Cncurrent Extractor ###############################
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     names = []
        #     for img in imgs:
        #         shm = multiprocessing.shared_memory.SharedMemory(create=True, size=img.nbytes)
        #         # Now create a NumPy array backed by shared memory
        #         s_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
        #         # print(img.shape, img.dtype)
        #         s_img[:] = img[:]  # Copy the original data into shared memory
        #         names.append(shm.name)
        #     t01 = time.time()

        #     results = executor.map(self.extract, names)
        #     for result in results:
        #         if result is not None:
        #             print(result[0].shape, result[0].dtype)
        #             print(result[1].shape, result[1].dtype)

        #             # self.features_cv.append([])
        #             self.descriptors.append(result[1])
        #             # self.features.append([Feature(queryKeypoints[i],queryDescriptors[i]) for i in range(len(queryKeypoints))])
        #             self.associations.append([False for _ in range(len(result[0]))])
        #             self.feature_pts.append(result[0])
        #         else:
        #             self.imgs.append(None)
        #             # self.features_cv.append([])
        #             self.descriptors.append([])
        #             self.feature_pts.append([])
        #             # self.features.append([])
        #             self.associations.append([])

        # self.imgs = imgs

    def extract(self, name):
        existing_shm = multiprocessing.shared_memory.SharedMemory(name=name)
        img = np.ndarray((3000, 4096, 3), dtype=np.uint8, buffer=existing_shm.buf)
        orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )
        # Note that a.shape is (6,) and a.dtype is np.int64 in this example
        # c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)
        if img is not None:
            img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            queryKeypoints, queryDescriptors = orb.detectAndCompute(img_gs, None)
            features = [
                np.array(cv_point.pt) for cv_point in queryKeypoints
            ]  # return must be pickle-able
            return [features, queryDescriptors]
        else:
            return None

    def filter_features(self, cam, uv, window_r):

        within_window = np.all(abs(self.feature_pts[cam] - uv) < window_r, axis=1)

        return within_window

    def makeKeyFrame(self):
        # Remove image
        self.imgs = None
