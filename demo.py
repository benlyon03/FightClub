import cv2
import numpy as np
import tensorflow as tf
import random

# Replace with the path to your saved model file
model = tf.keras.models.load_model('fightnight_iter1.h5')

# Replace with the path to your video file
fight_path = './predict_data/short_fight.mp4'
dance_path = 'predict_data/short_dance.mp4'


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
    return 'Violent Video'


# predicting whether the fight video is violent
single_vid = frames_from_video_file(fight_path, 8)
single_vid = tf.expand_dims(single_vid, axis=0)
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(single_vid)
# here the result of pooling features
print(predictions[0])
# here is what the computer is guessing the answer is
print(numToVal(np.argmax(predictions[0])))
print()
print()


single_vid = frames_from_video_file(dance_path, 8)
single_vid = tf.expand_dims(single_vid, axis=0)
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(single_vid)
print(predictions[0])
print(numToVal(np.argmax(predictions[0])))
