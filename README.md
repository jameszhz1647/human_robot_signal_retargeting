# human_robot_signal_retargeting

* Haozhi Zhang
* MSR Final Project  
* [Portfolio Post](https://aonai.github.io/linked_posts/myo_robot_arm.html)
* Human Data Acquisition
* Robot Data Acquisition

## Overview

The goal of this project is to mapping human motion to robot motion using a cross-domain multimodal variational autoencoder (mVAE) model. The human motion contains two type of electronic signal: electromyography (EMG) and inertial measurement unit (IMU), collecting by Myo arm band. The robot motion contains the positions and velocities of 7 joints of Franka pandas arm, collecting by the simulation in ROS MoveIt. This project applies signal processing and data augmentation on the acquisited huamn/robot data to build the dataset. The mVAE model is adpated from the paper [Multimodal representation models for prediction and control from partial information](https://www.sciencedirect.com/science/article/pii/S0921889019301575).

## Pipeline

![pipeline](source/pipeline.png)

## Folders

1. raw_data:

   1. human_: including 5 tasks, each task contains 2 EMG and 2 IMU signals, corresponding to upper-right arm and lower-right arm.

   2. robot_: including 5 sub folders: each folder corresponding to one task, contains 4 segments, each segment has positions and  velocities of 7 joints.

2. data_processing:

   1. post_processing: singal post-processing for both raw data collected from human and robot. 
      1. For human: downsampling (decimate) both EMG/IMU to the rate at 10 Hz
      2. For robot: downsampling (decimate) one segment of datapoints to 8 points, add 2 static points at the end of each segment. Integrate 4 segments into single sequence. 
      3. Concatenate 10 repeats for both human and robot data, and then align them to obtain the original dataset.
   2. data_augmentation: apply augmentaion on original dataset in order to train the mVAE model
      1. Using sklearn.preprocessing.MinMaxScaler to normailize the original data to the range at [-1, 1]
      2. Horizontally concatenate all data points at current time (at t) with them at previous time (at t -1)
      3. Split dataset into training set and testing set at ratio of 80:20
      4. Mask all robot data in training set with value -2 to obtain the case 2 dataset; mask all original data at t in training set with value -2  to obtain case 3 dataset
      5. Vertically concatenate original data with case 2 and case 3 data to obtain the final augmented training set