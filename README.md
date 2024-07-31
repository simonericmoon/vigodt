# VIGODT: Visual Interface for Georeferencing and Object Detection and Tracking

## Overview

VIGODT is a powerful web-based application that combines video processing, object detection and tracking, and geospatial visualization. It allows users to upload and process videos, perform object detection, and visualize the results on an interactive map. This tool is particularly useful for analyzing drone footage or any geo-tagged video content.

This object tracking system implements a simple yet effective method for maintaining consistent object identities across video frames. The core of this system is based on the Intersection over Union (IOU) metric and a track management mechanism.
The IOU metric quantifies the overlap between bounding boxes, providing a basis for associating detections with existing tracks. Each track maintains an 'age' parameter, incremented when unmatched and reset upon successful matching. This age-based approach allows the system to handle temporary occlusions and missed detections.
Two key parameters govern the tracking behavior:

iou_threshold: Defines the minimum overlap required to associate a detection with an existing track.
max_age: Determines how long a track can persist without matching before being removed.

For each frame, new detections are compared against existing tracks. Matches are established based on class consistency and IOU values exceeding the threshold. Unmatched detections spawn new tracks, while existing tracks are updated or removed based on their age.
<img src="images/Screenshot 2024-07-25 082040.png">
<img src="images/Screenshot 2024-07-25 082119.png">
<img src="images/Screenshot 2024-07-25 082235.png">
## Features

- User Authentication: Secure login system to protect your data.
- Video Processing: Upload and process MP4 video files.
- Object Detection: Utilizes YOLO (You Only Look Once) algorithm for efficient object detection.
- Geospatial Visualization: Interactive map display of detected objects.
- Time Range Selection: Choose specific segments of video for processing.
- Frame-by-Frame Analysis: Slider to view detected objects at specific frames.
- Real-time Map Updates: Map synchronizes with video playback (when "Show all frames" is unchecked).

## Requirements

- Python 3.7+
- Streamlit
- OpenCV (cv2)
- PyTorch
- Ultralytics YOLO
- Folium
- Streamlit-Folium
- PyYAML
- Streamlit-Authenticator

## Usage

1. switch branch "modularity", clone repo, build and start the docker-compose
2. Open a web browser and navigate to the provided local URL (typically `http://localhost:8501`).
3. Log in using your credentials.
4. Navigate through the file system to select an MP4 video file.
5. If an SRT file is present, you can select a time range for processing.
6. Click "Process Video" to start object detection and georeferencing.
7. Once processing is complete, the video will appear in the middle column and the map in the right column.
8. Use the slider and checkbox under the map to control the frame display.
9. If "Show all frames" is unchecked, the map will update as you play the video.

## Configuration

- Adjust the `base_dir` variable in the script to set the root directory for video browsing.
- Modify the `icons` dictionary to customize map markers for different object classes.
- Update the `SERVER_URL` if you're using a different FROST server for data upload.

## Notes

- Ensure that your videos have corresponding SRT files for georeferencing to work correctly.
- Processing time depends on the video length and complexity. Be patient with larger files.
- The application requires a GPU for optimal performance, especially for longer videos.

## Contributing

Contributions to VIGODT are welcome! Please feel free to submit pull requests, create issues or spread the word.

## Contact

Simon Eric Korfmacher
simon.korfmacher@iosb-fraunhofer.de
IOSB - SIRIOS

For any questions, issues, or contributions, please open an issue on the gitlab repository.