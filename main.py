
#import datetime
from pandas import datetime
from datetime import datetime
import pandas as pd
import threading
import queue
import os
import time


from prediction import *
from detection import *

def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return pd.to_datetime(new_time, format='%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except ValueError:
        return pd.to_datetime(new_time, format='%Y-%m-%d')  # # for total bus current data
    '''
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except:
        return datetime.strptime(new_time, '%Y-%m-%d')  # for total bus current data
    '''
# Queue to handle the scores of the predictions
queue_score = queue.Queue()
datasets = ['WheelTemperature.csv', 'WheelRPM.csv']
var_names = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Temperature (C)', 'RPM']

# Function to stream input data into the predictor
def stream_inputData():
    global x_event
    global fileEnd
    global ds_name
    for ds in range(len(datasets)):
        fileEnd = False
        dataset = datasets[ds] 
        ds_name = dataset[:-4]  #drop 'Data/' and '.csv' (Extract dataset name (e.g., 'BusVoltage'))
        print('ds_name:', ds_name)
        dataset_path = 'E:/Satellite-Telemetry-Anomaly-Detection-master/Data/' + dataset
        print('Path:', dataset_path)
        # Load and preprocess dataset
        ts = pd.read_csv(dataset_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
        ts.fillna(value=0, inplace=True) # Handle missing values
        print('ts len: ', len(ts))
        # Initialize the predictor
        predictor = rrcf_stream(ds_name, num_trees=100, shingle_size=18, tree_size=256) 

        for index, value in ts.items():
            try:
                # Process each predicted_val using the existing RRCF logic     
                predicted_val = predictor.score_with_rrcf(index, value)
                if predicted_val is not None: 
                    queue_score.put(predicted_val) # Add score to the queue
            except Exception as e:
                print(f"Error processing predicted_val at {index}: {e}")
        fileEnd = True
        print('fileEnd predictor: ', fileEnd)
        x_event.wait()

# Run the input data stream in a separate thread       
predictor_thread = threading.Thread(target=stream_inputData)
predictor_thread.start()
x_event = threading.Event()
# -------------------------------------------------------------------------------------------------------------
# Initialize the anomaly detector
outlier_def='codisp'
num_stds=[2,4,8]
# Function to process scores and detect anomalies
def stream_scoreData():
    global x_event
    global fileEnd
    global ds_name
    while predictor_thread.is_alive(): #new dataset
        print('ds_name predictor: ', ds_name)
        data = []
        detector0 = OnlineAnomalyDetector(ds_name, outlier_def, num_stds[0])
        detector1 = OnlineAnomalyDetector(ds_name, outlier_def, num_stds[1])
        detector2 = OnlineAnomalyDetector(ds_name, outlier_def, num_stds[2])

        while predictor_thread.is_alive() or not queue_score.empty():
            try:
                if not queue_score.empty():
                    score = queue_score.get()                    
                    data.append(score[2])
                    codisps_array = np.array(data)
                    # Compute the standard deviation with ddof=1 for sample
                    std_codisp = float(codisps_array.std(ddof=1)) #ddof=0 : Population std (offline)
                    mean_codisp = np.mean(codisps_array)
                    detector0.detect_anomaly(score, std_codisp, mean_codisp)
                    detector1.detect_anomaly(score, std_codisp, mean_codisp)
                    detector2.detect_anomaly(score, std_codisp, mean_codisp)
                    
                elif fileEnd == True:
                    break
            except Exception as e:
                print(f"Error detecting anomaly: {e}")
        time.sleep(5)
        print('fileEnd detector: ', fileEnd)
        x_event.set()
        x_event = threading.Event()
        print('fileEnd detector: ', fileEnd)
        time.sleep(5)
        print('ds_name: ', ds_name)

# Run the scoring thread
detector_thread = threading.Thread(target=stream_scoreData)
detector_thread.start()

# Ensure threads finish before exiting the program
predictor_thread.join()
detector_thread.join()

print("Processing completed.")
os.system("shutdown /s /t 360")