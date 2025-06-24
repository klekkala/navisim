import cv2
import os

# image_folder = './data/'
# video_name = 'output_video'
# fps = 2

def images2video(input_dir, output_dir, output_name, fps=30, output_format="mp4v"):
    images = [img for img in os.listdir(input_dir) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Sort to maintain order
    
    frame = cv2.imread(os.path.join(input_dir, images[0]))
    height, width, layers = frame.shape

    # Setting the output format to .mp4
    full_output_path = os.path.join(output_dir, f"{output_name}.mp4")
    
    # Using 'mp4v' as the codec for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*output_format)

    video = cv2.VideoWriter(full_output_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_dir, image)))

    video.release()
    cv2.destroyAllWindows()
