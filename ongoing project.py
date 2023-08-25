import cv2
import numpy as np

# Open the video file
video_path = 'path'
cap = cv2.VideoCapture(video_path)

frames = []
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Append the frame to the frames list
    frames.append(frame)
    frame_count += 1

    # Display the current frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Convert the frames list into a 3D NumPy array (3D matrix)
video_matrix = np.array(frames)

# Print the dimensions of the video matrix
print("Video Matrix Shape:", video_matrix.shape)


new_video_path = 'path_to_new_video_file.mp4'
new_cap = cv2.VideoCapture(new_video_path)

new_frames = []
frame_count = 0

# Loop through the frames of the new video
while new_cap.isOpened():
    ret, new_frame = new_cap.read()
    
    if not ret:
        break
    
    new_frames.append(new_frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

new_cap.release()
cv2.destroyAllWindows()

# Convert new frames to a 3D matrix
new_video_matrix = np.array(new_frames)

# Compare the two video matrices using Mean Squared Error (MSE)
mse = np.mean(np.square(previous_video_matrix - new_video_matrix))

print(f"Mean Squared Error between videos: {mse}")