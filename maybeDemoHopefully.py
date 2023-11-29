import cv2
import time

def record_video(output_path, duration):
    # Open a video capture object (0 represents the default camera)
    cap = cv2.VideoCapture(0)

    # Get the width and height of the frames
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
    output_mp4_path = r'C:\Users\2alex\OneDrive\Documents\GitHub\FightClub\liveVideo.mp4'
    recording_duration = 4  # in seconds

    # Start recording
    record_video(output_avi_path, recording_duration)

    # Convert AVI to MP4
    convert_avi_to_mp4(output_avi_path, output_mp4_path)