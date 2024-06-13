# monoslam-cal

Implementation of using a SLAM formulation to calibrate monocular cameras in real-time

## Maintainers

Jose A. Medina - josem419@stanford.edu

## Motivation

This code was implmented as part of a university project. My goal was to get familiarity with visual SLAM systems by implementing a basic one. 

The goal is to use a SLAM system and techniques to calibrate cameras. While most SLAM systems assume a good camera extrinsic calibration as a prior, my goal was to leverage SLAM for online calibration on a mobile robot. 

The choice to use python was one of convenience given the available libraries. For a more performant real-time/online SLAM system for calibration I would reimplemnt this in C++. 

## Acknowledgements

A lot of credit goes to the following sources which were used a references throughout this project. 


Michael Burri, Janosch Nikolic, Pascal Gohl, Thomas
Schneider, Joern Rehder, Sammy Omari, Markus W Achte-
lik, and Roland Siegwart. The euroc micro aerial vehicle
datasets. The International Journal of Robotics Research, 2016. 2

[2] Carlos Campos, Richard Elvira, Juan J. Gomez, José M. M.
Montiel, and Juan D. Tardós. ORB-SLAM3: An accurate
open-source library for visual, visual-inertial and multi-map
SLAM. IEEE Transactions on Robotics, 37(6):1874–1890, 2021. 6

[3] Gerardo Carrera, Adrien Angeli, and Andrew J. Davison.
Slam-based automatic extrinsic calibration of a multi-camera
rig. In 2011 IEEE International Conference on Robotics and
Automation, pages 2652–2659, 2011. 1

[4] Javier Civera, Diana R. Bueno, Andrew J. Davison, and
J. M. M. Montiel. Camera self-calibration for sequential
bayesian structure from motion. In 2009 IEEE International
Conference on Robotics and Automation, pages 403–408, 2009. 1, 2

[5] Andrew J. Davison, Ian D. Reid, Nicholas D. Molton, and
Olivier Stasse. Monoslam: Real-time single camera slam.
IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, 29(6):1052–1067, 2007. 1, 2

[6] Frank Dellaert and GTSAM Contributors. borglab/gtsam,
May 2022. 7

[7] Martin A. Fischler and Robert C. Bolles. Random sample
consensus: A paradigm for model fitting with applications
to image analysis and automated cartography. In Martin A.
Fischler and Oscar Firschein, editors, Readings in Computer
Vision, pages 726–740. Morgan Kaufmann, San Francisco
(CA), 1987. 1, 4

[8] Richard Hartley and Andrew Zisserman. Multiple View Ge-
ometry in Computer Vision. Cambridge University Press,
New York, NY, USA, 2 edition, 2003. 2, 4

[9] Rainer Kümmerle, Giorgio Grisetti, Hauke Strasdat, Kurt
Konolige, and Wolfram Burgard. G2o: A general framework
for graph optimization. In 2011 IEEE International Confer-
ence on Robotics and Automation, pages 3607–3613, 2011.
4

[10] Tony Lindeberg. Scale Invariant Feature Transform, vol-
ume 7. 05 2012. 3

[11] David G Lowe. Distinctive image features from scale-
invariant keypoints. International journal of computer vi-
sion, 60:91–110, 2004. 3

[12] “Morgan Quigley, Brian Gerkey, Ken Conley, Josh Faust,
Tully Foote, Jeremy Leibs, Eric Berger, Rob Wheeler, and
Andrew Ng”. “ros: an open-source robot operating system”.
In “Proc. of the IEEE Intl. Conf. on Robotics and Automa-
tion (ICRA) Workshop on Open Source Robotics”, “Kobe,
Japan”, May 2009. 2

[13] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary
Bradski. Orb: An efficient alternative to sift or surf. In 2011
International Conference on Computer Vision, pages 2564–
2571, 2011. 3

[14] Jianbo Shi and Tomasi. Good features to track. In 1994
Proceedings of IEEE Conference on Computer Vision and
Pattern Recognition, pages 593–600, 1994. 3

[15] Pangolin: https://github.com/stevenlovegrove/Pangolin

[16] TwitchSlam: https://github.com/geohot/twitchslam 

[17] PySlam: https://github.com/luigifreda/pyslam 

[18] OpenCV: https://opencv.org/ 