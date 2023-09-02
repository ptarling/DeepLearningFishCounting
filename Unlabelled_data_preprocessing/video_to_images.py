import cv2
import os
import glob

save_interval = 12
# Define input video file path and output image save path
work_folder = os.getcwd()
video_folder = os.path.join(work_folder, "data/")

video_paths = glob.glob(video_folder + "2018*")
output_folder = os.path.join(
    work_folder, "Unlabelled_data_preprocessing/unlabel_images")

# Create a folder to save the images
os.makedirs(output_folder, exist_ok=True)

for i, video_path in enumerate(video_paths):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 1

    # Loop to read video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Generate the output image filename, for example: frame_0001.jpg
        output_filename = f'{os.path.splitext(os.path.basename(video_path))[0]}#{str(frame_count).zfill(4)}.jpg'
        output_path = os.path.join(output_folder, output_filename)

        # Save the image file
        if (frame_count % save_interval == 0):
            cv2.imwrite(output_path, frame)

        frame_count += 1
    print(f"converting the {i+1}th video....")

# Release the video object and close the window (if any)
cap.release()
cv2.destroyAllWindows()
