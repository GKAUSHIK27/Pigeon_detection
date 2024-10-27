Codes are in src folder

To train Run train.py #change .yaml path

To make predictions on a video Run predict.py or run this code
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("weights/best.pt")

# Open the video file
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

# Define output video parameters
output_video_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define codec for .mp4 format
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

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

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

