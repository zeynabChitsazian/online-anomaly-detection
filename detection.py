import numpy as np
from queue import Queue
from datetime import datetime

class OnlineAnomalyDetector:
    def __init__(self, ds_name, outlier_def, num_stds):
        self.ds_name = ds_name
        self.outlier_def = outlier_def
        self.num_stds = num_stds
        self.queue = Queue()
        self.results = []
        self.running = False
        self.current_obses = []
        self.count_anomaly = 0
        self.count_sample = 0
        self.oldTime = datetime.now()
        self.error_points = []
        self.error_indices = []
        self.codisps = []
        self.logFile = 'log_' + ds_name + '_' + outlier_def + str(num_stds) + '.txt'

    def detect_anomaly(self, data, std_codisp, mean_codisp):
        index = data[0]
        sample = data[1]
        predicted = data[2]
        if self.outlier_def == 'std':
            self.current_obses.append(sample)  # Add new observation
            obses_array = np.array(self.current_obses)
            
            # Compute the standard deviation with ddof=1 for sample
            self.std = float(obses_array.std(ddof=1)) #ddof=0 : Population std (offline)

            error = abs(predicted - sample)
            is_anomaly = error > self.std * self.num_stds
            self.count_sample += 1
            if is_anomaly is True:
                self.count_anomaly += 1
            if self.count_sample%100 == 0:
                with open(self.logFile, "a") as log_file:
                    log_file.write(f'sample: {sample}\n')
                    log_file.write(f'predicted: {predicted}\n')
                    log_file.write(f'error: {error}\n')
                    log_file.write(f'self.std * self.num_stds: {self.std * self.num_stds}\n')
                    log_file.write(f'current count_anomaly: {self.count_anomaly}\n')
                    log_file.write(f'current count_sample: {self.count_sample}\n')

                    elapsed_time = datetime.now() - self.oldTime
                    elapsed_time_seconds = elapsed_time.total_seconds()
                    elapsed_time_minutes = elapsed_time_seconds / 60

                    log_file.write(f"Elapsed time: {elapsed_time_seconds:.2f} seconds\n")
                    log_file.write(f"Elapsed time: {elapsed_time_minutes:.2f} minutes\n")
                    if is_anomaly: log_file.write(f'is_anomaly: {is_anomaly}\n')
                    log_file.write(f'----------------------------------------------------------\n\n\n')
                self.oldTime = datetime.now()
        elif self.outlier_def == 'codisp':
            self.count_sample += 1
            timeIndex = data[0]
            codisp = data[2]
            '''
            self.codisps.append(codisp)
            codisps_array = np.array(self.codisps)
            # Compute the standard deviation with ddof=1 for sample
            std_codisp = float(codisps_array.std(ddof=1)) #ddof=0 : Population std (offline)
            mean_codisp = np.mean(codisps_array)
            '''
            threshold = mean_codisp + self.num_stds * std_codisp
            is_anomaly = codisp > threshold
            if is_anomaly:
                self.count_anomaly += 1
            if self.count_sample%100 == 0:
                with open(self.logFile, "a") as log_file:
                    log_file.write(f'codisp: {codisp}\n')
                    log_file.write(f'threshold: {threshold}\n')
                    log_file.write(f'current count_anomaly: {self.count_anomaly}\n')
                    log_file.write(f'current count_sample: {self.count_sample}\n')

                    elapsed_time = datetime.now() - self.oldTime
                    elapsed_time_seconds = elapsed_time.total_seconds()
                    elapsed_time_minutes = elapsed_time_seconds / 60

                    log_file.write(f"Elapsed time: {elapsed_time_seconds:.2f} seconds\n")
                    log_file.write(f"Elapsed time: {elapsed_time_minutes:.2f} minutes\n")
                    if is_anomaly: log_file.write(f'is_anomaly: {is_anomaly}\n')
                    log_file.write(f'----------------------------------------------------------\n\n\n')
                self.oldTime = datetime.now()
        '''
        result = {
            "index": index,
            "observed": sample,
            "predicted": predicted,
            "error": error,
            "is_anomaly": is_anomaly
        }
        '''