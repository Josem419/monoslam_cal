#!/usr/bin/env python3

# Jose A Medina
# josem419@stanford.edu

import cv2
import numpy as np
import matplotlib.pyplot as plt



class Visualizer:

    def __init__(self) -> None:
        # self.data_manager = DataManager()
        pass

    def draw_features(self,img,features,col, size=10, thickness=10):
        """
        Draws feaures in the image. Features can be a list of cv2.KeyPoint or 2D list/np array
        """
        # return cv2.drawKeypoints(img,features,None, (0,0,255),flags=0)
        if len(features) == 0:
            return img
        if isinstance(features[0], cv2.KeyPoint):
            for curKey in features:
                x=np.int(curKey.pt[0])
                y=np.int(curKey.pt[1])
                cv2.circle(img,(x,y),size, col,thickness=thickness, lineType=8, shift=0)
        else:
            for curKey in features:
                x=np.int(curKey[0])
                y=np.int(curKey[1])
                cv2.circle(img,(x,y),size, col,thickness=thickness, lineType=8, shift=0)
        return img