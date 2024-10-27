import cv2
import numpy as np
from ultralytics import YOLO

def process_video(input_video_path, output_video_path, model):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        raise ValueError("Error opening video file.")

    # Define output video parameters
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    try:
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLO inference on the frame with confidence threshold
                results = model(frame, conf=0.6)  # Set confidence threshold here
                
                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Write the annotated frame to the output video
                out.write(annotated_frame)

                # Display the annotated frame (optional)
                cv2.imshow("YOLO Inference", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Example usage:
# model = YOLO('path/to/your/model.pt')  # Load your YOLO model here
# process_video("input_video.mp4", "output_video.mp4", model)
