import gtsam
import numpy as np
from data_manager import DataManager


class IMU:

    def __init__(self, previous_timestamp: float, timestamp: float, accumulated_imu, frame_id_corr: int) -> None:
        
        self.data_manager = DataManager()
        
        self.frame_id_corr = frame_id_corr
        self.data = accumulated_imu

        # TODO create method for this, as it change based on which frames are being considered
        # e.g. need a way to sum IMU factors
        self.summed_imu_z = gtsam.PreintegratedImuMeasurements(self.data_manager.config.imu_params, gtsam.imuBias.ConstantBias())
        if len(accumulated_imu) > 0:
            t_prev_imu = previous_timestamp
            dt_imu = None
            for j in range(len(accumulated_imu)):
                # Calculate dt:
                if t_prev_imu is not None: 
                    if j==len(accumulated_imu)-1:
                        dt_imu = timestamp - t_prev_imu  # last one up until image time
                    else:
    
                        time_imu = accumulated_imu[j][0]
                        dt_imu = time_imu - t_prev_imu
                        t_prev_imu = time_imu

                # if its still none it may be because we havent initialized well
                # default to a known dt
                if dt_imu is None:
                    dt_imu = 0.05 # this is the known rate of the imu
                
                # angluar velocity, linear acceleration, and dt
                self.summed_imu_z.integrateMeasurement(
                    accumulated_imu[j][1][3:],
                    accumulated_imu[j][1][:3],
                    dt_imu
                )
        

