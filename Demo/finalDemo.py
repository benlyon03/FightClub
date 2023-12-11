
import cv2
import numpy as np
import tensorflow as tf
import random
import time
import sys
sys.path.append('mmAction/mmaction2/')
from mmaction.apis import init_recognizer, inference_recognizer
import os
import sklearn.metrics as metrics


def format_frames(frame, output_size):
    """
      Pad and resize an image from a video.

      Args:
        frame: Image that needs to resized and padded.
        output_size: Pixel size of the output frame image.

      Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(fight_path, n_frames, output_size=(224, 224), frame_step=15):
    """
      Creates frames from each video file present for each category.

      Args:
        fight_path: File path to the video.
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

      Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(fight_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


# NV = 0, V = 1
def numToVal(num):
    if num == 0:
        return 'Non Violent Video'
    elif num == 1:
        return 'Violent Video'
    return 'Some Very Wrong Is Happening'


def convert_avi_to_mp4(avi_path, mp4_path, fps=30):
    # Open the AVI file
    cap = cv2.VideoCapture(avi_path)

    # Get the width and height of the frames
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Define the codec and create a VideoWriter object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Write the frame to the output MP4 file
        out.write(frame)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

if __name__ == "__main__":
    # Set the output video path and recording duration

    # Input video paths for the models
    inp = ['Demo\Videos\boxing.mp4', 'Demo\Videos\brawl.mp4']

    while(True):

        for vid in inp:
            play =  cv2.VideoCapture(vid)
        # Load in model
            while True:
                # Read a frame from the video
                ret, frame = play.read()

                # Check if the video has ended
                if not ret:
                    print("Video playback completed.")
                    break
                # font 
                font = cv2.FONT_HERSHEY_SIMPLEX 
                
                # org 
                org = (50, 50) 
                
                # fontScale 
                fontScale = 1
                
                # White color in BGR 
                color = (255, 255, 255) 
                
                # Line thickness of 2 px 
                thickness = 2
                
                # Using cv2.putText() method 
                # frame = cv2.putText(frame, "Non-violent" if results.pred_score[0] > results.pred_score[1] else "Violent", org, font,  
                #                 fontScale, color, thickness, cv2.LINE_AA)
                # Display the frame (you can perform additional processing here)
                cv2.imshow('Video', frame)

                # Exit the loop if 'q' key is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            play.release()

            # Loads model from fightnight_iter1.h5. Change it to the path
            model = tf.keras.models.load_model('fightnight_iter1.h5')

            # Preprocess input
            single_vid = frames_from_video_file(vid, 8)
            single_vid = tf.expand_dims(single_vid, axis=0)
            probability_model = tf.keras.Sequential([model,
                                                    tf.keras.layers.Softmax()])
            predictions = probability_model.predict(single_vid)

            # Load the newly trained model checkpoint
            # Change it to the paths to epoch_10.pth and config.py
            new_checkpoint = 'epoch_10.pth'
            cfg = 'config.py'

            # Initialize the recognizer with the new checkpoint
            new_model = init_recognizer(cfg, new_checkpoint, device='cuda:0')

            # Specify the label map
            #video = "nofight.mp4"
            results = inference_recognizer(new_model, vid)

            print("-------------------------------------------------------------------\n")
            
            # EfficientNet Results
            print("EfficientNet Results:")
            print("Predictions Scores: ", [predictions[0][0], predictions[0][1]])
            print("Non-violent") if predictions[0][0] > predictions[0][1] else print("Violent")
            print("-------------------------------------------------------------------\n")

            # MMAction Results
            print("MMAction Results:")
            print("Prediction Scores: ", results.pred_score)
            print("Non-violent") if results.pred_score[0] > results.pred_score[1] else print("Violent")
            print("-------------------------------------------------------------------\n")
            
            #Create video player to play video
            play =  cv2.VideoCapture(vid)

            while True:
                # Read a frame from the video
                ret, frame = play.read()

                # Check if the video has ended
                if not ret:
                    print("Video playback completed.")
                    break
                # font 
                font = cv2.FONT_HERSHEY_SIMPLEX 
                
                # org 
                org = (50, 50) 
                
                # fontScale 
                fontScale = 1
                
                # White color in BGR 
                color = (255, 255, 255) 
                
                # Line thickness of 2 px 
                thickness = 2
                
                # Using cv2.putText() method 
                frame = cv2.putText(frame, "Non-violent" if results.pred_score[0] > results.pred_score[1] else "Violent", org, font,  
                                fontScale, color, thickness, cv2.LINE_AA)
                # Display the frame (you can perform additional processing here)
                cv2.imshow('Video', frame)

                # Exit the loop if 'q' key is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Release the VideoCapture object and close all windows
            play.release()
            cv2.destroyAllWindows()