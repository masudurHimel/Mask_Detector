import tensorflow as tf
import cv2
import numpy as np
import os
import skvideo.io
from os.path import isfile, join
from google.colab.patches import cv2_imshow
import dlib
import math
import sys
import pickle
import argparse


def requiredPackage():
  try:
    ! pip install colorama==0.3.9
    ! pip install imutils==0.5.1
    ! pip install progress==1.4
    ! pip install sk-video==1.1.10
    ! pip install moviepy
    ! pip install speechpy
    ! pip install pysoundfile
    print('Successful')
  except:
    print("Failed")

requiredPackage()



def maskDetector(inputPath, outputPath, fps =30, codecType = "MJPG"):
    inputPath = inputPath
    outputPath = outputPath
    fps = fps
    codecType = codecType

    temp = outputPath
    print(temp)
    temp = temp.split("/")
    temp = temp[-1].split(".")
    temp = temp[0]
    
    


    # Dlib requirements.
    predictor_path = 'dlib/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    mouth_destination_path = os.path.dirname(outputPath) + '/' + 'mouth' + '/' + temp
    if not os.path.exists(mouth_destination_path):
        os.makedirs(mouth_destination_path)


    inputparameters = {}
    outputparameters = {}
    reader = skvideo.io.FFmpegReader(inputPath,
                    inputdict=inputparameters,
                    outputdict=outputparameters)
    video_shape = reader.getShape()
    (num_frames, h, w, c) = video_shape
    print(num_frames, h, w, c)

    # The required parameters
    activation = []
    max_counter = 150
    total_num_frames = int(video_shape[0])
    num_frames = min(total_num_frames,max_counter)
    counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Define the writer
    writer = skvideo.io.FFmpegWriter(outputPath)


    # Required parameters for mouth extraction.
    width_crop_max = 0
    height_crop_max = 0


    # Loop over all frames.
    for frame in reader.nextFrame():
        ##print('frame_shape:', frame.shape)

        # Process the video and extract the frames up to a certain number and then stop processing.
        if counter > num_frames:
            break

        # Detection of the frame
        frame.setflags(write=True)
        detections = detector(frame, 1)

        # 20 mark for mouth
        marks = np.zeros((2, 20))

        # All unnormalized face features.
        Features_Abnormal = np.zeros((190, 1))

        # If the face is detected.
        ##print(len(detections))
        if len(detections) > 0:
            for k, d in enumerate(detections):

                # Shape of the face.
                shape = predictor(frame, d)

                co = 0
                # Specific for the mouth.
                for ii in range(48, 68):
                    X = shape.part(ii)
                    A = (X.x, X.y)
                    marks[0, co] = X.x
                    marks[1, co] = X.y
                    co += 1

                # Get the extreme points(top-left & bottom-right)
                X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),
                                                    int(np.amax(marks, axis=1)[0]),
                                                    int(np.amax(marks, axis=1)[1])]

                # Find the center of the mouth.
                X_center = (X_left + X_right) / 2.0
                Y_center = (Y_left + Y_right) / 2.0

                # Make a boarder for cropping.
                border = 30
                X_left_new = X_left - border
                Y_left_new = Y_left - border
                X_right_new = X_right + border
                Y_right_new = Y_right + border

                # Width and height for cropping(before and after considering the border).
                width_new = X_right_new - X_left_new
                height_new = Y_right_new - Y_left_new
                width_current = X_right - X_left
                height_current = Y_right - Y_left

                # Determine the cropping rectangle dimensions(the main purpose is to have a fixed area).
                if width_crop_max == 0 and height_crop_max == 0:
                    width_crop_max = width_new
                    height_crop_max = height_new
                else:
                    width_crop_max += 1.5 * np.maximum(width_current - width_crop_max, 0)
                    height_crop_max += 1.5 * np.maximum(height_current - height_crop_max, 0)

                
                X_left_crop = int(X_center - width_crop_max / 2.0)
                X_right_crop = int(X_center + width_crop_max / 2.0)
                Y_left_crop = int(Y_center - height_crop_max / 2.0)
                Y_right_crop = int(Y_center + height_crop_max / 2.0)
                

                if X_left_crop >= 0 and Y_left_crop >= 0 and X_right_crop < w and Y_right_crop < h:
                    mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]

                    # Save the mouth area.
                    mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                    
                    cv2.imwrite(mouth_destination_path + '/' + 'frame' + '_' + str(counter) + '.png', mouth_gray)

                    ##print("The cropped mouth is detected ...")
                    activation.append(1)
                else:
                    cv2.putText(frame, 'The full mouth is not detectable. ', (30, 30), font, 1, (0, 255, 255), 2)
                    print("The full mouth is not detectable. ...")
                    activation.append(0)

        else:
            cv2.putText(frame, 'Mask On !', (130, 130), font, 3, (0, 255, 0), 4)
            #print("Mouth is not detectable. ...")
            activation.append(0)


        if activation[counter] == 1:
            # Demonstration of face.
            #cv2.rectangle(frame, (X_left_crop, Y_left_crop), (X_right_crop, Y_right_crop), (0, 255, 0), 2)
            cv2.putText(frame, 'Mask Off !!', (130, 130), font, 3, (255, 0, 0), 4)

        # cv2.imshow('frame', frame)
        ##print('frame number %d of %d' % (counter, num_frames))

        # write the output frame to file
        ##print("writing frame %d with activation %d" % (counter + 1, activation[counter]))
        writer.writeFrame(frame)
        counter += 1

    writer.close()



    the_filename = os.path.dirname(outputPath) + '/' + 'activation'
    my_list = activation
    with open(the_filename, 'wb') as f:
        pickle.dump(my_list, f)

    ##print("Completed -- ", inputPath)

maskDetector(inputPath = "TestVideos/Another/Output/input2.mp4", outputPath = "TestVideos/Another/Output/output3.mp4")

