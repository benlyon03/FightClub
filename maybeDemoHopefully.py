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


def record_video(output_path, duration):
    # Open a video capture object (0 represents the default camera)
    cap = cv2.VideoCapture(0)

    # Get the width and height of the frames
    # Dataset dimensions: 244 x 244
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    recording = False
    start_time = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Recording...', frame)

        # Check for 'r' key press to start/stop recording
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                print("Recording started.")
                start_time = time.time()
                recording = True
            else:
                print("Recording stopped.")
                recording = False

        # Check if recording and within the specified duration
        if recording and time.time() - start_time < duration:
            # Write the frame to the output video
            out.write(frame)
        elif recording:
            print("Recording reached the specified duration.")
            recording = False

        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


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
    output_avi_path = r'C:\Users\2alex\OneDrive\Documents\GitHub\FightClub\liveVideo.avi'
    output_mp4_path = r'C:\Users\2alex\OneDrive\Documents\GitHub\FightClub\dance.mp4'
    recording_duration = 4  # in seconds

    # Start recording
    #record_video(output_avi_path, recording_duration)

    # Convert AVI to MP4
    #convert_avi_to_mp4(output_avi_path, output_mp4_path)

    # Load in model
    model = tf.keras.models.load_model('fightnight_iter1.h5')

    # Replace with the path to your video file
    input = r"C:\Users\2alex\Downloads\Climate protester run over by lorry in Germany.mp4"

    # Preprocess input
    single_vid = frames_from_video_file(output_mp4_path, 8)
    single_vid = tf.expand_dims(single_vid, axis=0)
    probability_model = tf.keras.Sequential([model,
                                            tf.keras.layers.Softmax()])
    predictions = probability_model.predict(single_vid)

    # The result of pooling features
    print("EfficientNet Results:")
    print(predictions[0][0], predictions[0][1])

    # here is what the computer is guessing the answer is
    print("Non-violent") if predictions[0][0] > predictions[0][1] else print("Violent")
    print("-------------------------------------------------------------------\n")

    # Load the newly trained model checkpoint
    new_checkpoint = 'epoch_10.pth'
    cfg = 'config.py'

    # Initialize the recognizer with the new checkpoint
    new_model = init_recognizer(cfg, new_checkpoint, device='cuda:0')

    # Specify the label map
    #video = "nofight.mp4"
    results = inference_recognizer(new_model, output_mp4_path)

    print("MMAction Results:")
    print(results.pred_score)
    