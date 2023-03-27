import cv2
import os

# Directory where your frames are stored
frame_dir = './video'
video_name = 'mandelbrot_animation'

# Set the frame rate of the output video
fps = 60

# Define the codec for the output video (can be changed based on the desired output format)
fourcc = cv2.VideoWriter_fourcc(*'H264')

# Set the size of the output video (must match the size of your frames)
frame_size = (3840, 2160)

# Create a VideoWriter object
output_video = cv2.VideoWriter(f'{frame_dir}/{video_name}.mp4', fourcc, fps, frame_size)

# Loop through the frames and add them to the output video
from tqdm import tqdm
for i in tqdm(range(len(os.listdir(frame_dir)))):
    # Load the frame
    frame = cv2.imread(os.path.join(frame_dir, f'mandelbrot_f{i}.png').replace('\\', '/'))

    # Write the frame to the output video
    output_video.write(frame)

# Release the VideoWriter object
output_video.release()
