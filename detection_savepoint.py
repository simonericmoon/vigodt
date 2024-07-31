"""
Script Name: Video Object Detection and Georeferencing
Description: This script performs object detection on video files and georeferences detected objects using associated SRT files. It supports outputting the results to a JSON file or directly uploading them to a specified server.
Author: Simon Eric Korfmacher
Date: 04.07.2024
Organization: Fraunhofer IOSB (Institute of Optronics, System Technologies and Image Exploitation) SIRIOS
Version: 1.0

Requirements:
- OpenCV (cv2)
- PyTorch (torch)
- NumPy (numpy)
- Ultralytics YOLO (for object detection)
- argparse
- re
- math
- json
- requests

Note:
- Ensure all required libraries are installed before running this script.
- You need to have the appropriate permissions to upload data to the server specified in the SERVER_URL variable.

Contact:
- For any questions or contributions, please contact simon.korfmacher@iosb.fraunhofer.de.
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import argparse
import re
from math import radians, cos, sin, asin, atan2, degrees
import json
import requests
import subprocess

# Define server URL for data upload
SERVER_URL = "https://sirios-frost-drohnen.k8s.ilt-dmz.iosb.fraunhofer.de/FROST-Server/v1.1/Things"

def find_srt_file(video_path):
    """
    Finds a .SRT or .srt file in the same directory as the video file.
    """
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for ext in ['.SRT', '.srt']:
        srt_path = os.path.join(video_dir, video_name + ext)
        if os.path.exists(srt_path):
            return srt_path
    
    return None

def parse_srt_file(video_path):
    srt_file_path = find_srt_file(video_path)
    if srt_file_path is None:
        print(f"Error: No corresponding SRT file found for {video_path}")
        return None

    frame_data_regex = re.compile(
        r'\d+\n'
        r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n'
        r'<font[^>]*>FrameCnt: (\d+),[^[]*'
        r'(?:\[iso:[^\]]*\])?\s*'  # Optional ISO setting
        r'(?:\[shutter:[^\]]*\])?\s*'  # Optional shutter setting
        r'(?:\[fnum:[^\]]*\])?\s*'  # Optional f-number setting
        r'(?:\[ev:[^\]]*\])?\s*'  # Optional EV setting
        r'(?:\[color_md :[^\]]*\])?\s*'  # Optional color mode setting
        r'(?:\[ae_meter_md:[^\]]*\])?\s*'  # Optional AE meter mode setting
        r'\[focal_len: ([\d.]+)\] \[dzoom_ratio: ([\d.]+)\], '
        r'\[latitude: ([\d.]+)\] \[longitude: ([\d.-]+)\] '
        r'\[rel_alt: ([\d.]+) abs_alt: [\d.]+\] '
        r'\[gb_yaw: ([\d.-]+) gb_pitch: ([\d.-]+) gb_roll: ([\d.-]+)\] </font>'
    )

    frames_info = {}
    with open(srt_file_path, 'r') as file:
        srt_content = file.read()
    for match in frame_data_regex.finditer(srt_content):
        frame_id = int(match.group(3))
        frames_info[frame_id] = {
            "focal_length_mm": float(match.group(4)),
            "dzoom_ratio": float(match.group(5)),
            "latitude": float(match.group(6)),
            "longitude": float(match.group(7)),
            "rel_alt": float(match.group(8)),
            "gb_yaw": float(match.group(9)),
            "gb_pitch": float(match.group(10)),
            "gb_roll": float(match.group(11)),
        }
    return frames_info

def video_object_detection_and_georeferencing(video_path, model_path, cls_idx=None, sensor_width_mm=6.4, output="json"):
    """
    Performs video object detection and georeferencing, then outputs the data based on the specified method.
    """
    if cls_idx is None:
        cls_idx = [0, 1, 2, 3, 5, 6, 7, 15, 16, 17]  # Default class indexes if not provided

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Used device: {device}")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    frames_info = parse_srt_file(video_path)
    if frames_info is None:
        return

def calculate_new_gps(lat, lon, distance, bearing):
    """
    Calculates new GPS coordinates given a starting point, distance, and bearing.
    """
    R = 6371e3  # Earth radius in meters
    bearing = radians(bearing)
    lat1 = radians(lat)
    lon1 = radians(lon)
    lat2 = asin(sin(lat1) * cos(distance / R) + cos(lat1) * sin(distance / R) * cos(bearing))
    lon2 = lon1 + atan2(sin(bearing) * sin(distance / R) * cos(lat1), cos(distance / R) - sin(lat1) * sin(lat2))
    return degrees(lat2), degrees(lon2)

def createThing(row):
    """
    Creates a Thing object for server upload, based on row data.
    """
    try:
        lat = float(row['lat'].replace(',', '.'))
        lon = float(row['lon'].replace(',', '.'))
        if lat is None or lon is None:
            print(f"Skipping row due to missing lat/lon: {row}")
            return None
    except ValueError as e:
        print(f"Skipping row due to invalid lat/lon: {row}")
        return None

    properties = {
        "Class": row.get('klasse', ''),
        "FrameID": row.get('frameid', ''),
    }

    known_fields = ['ID', 'klasse', 'lat', 'lon', 'frameid']
    for key, value in row.items():
        if key not in known_fields:
            properties[key] = value

    thing = {
        "name": "Objekt",
        "description": "Object in Drone scene",
        "properties": properties,
        "Locations": [{
            "name": "Object location",
            "description": "The location of the object",
            "encodingType": "application/vnd.geo+json",
            "location": {
                "type": "Point",
                "coordinates": [lat, lon],
            },
        }]
    }
    return thing

def upload_data_to_server(Things):
    """
    Uploads a list of Thing objects to the server.
    """
    url = SERVER_URL
    headers = {"Content-Type": "application/json"}
    for thing in Things:
        try:
            response = requests.post(url, data=json.dumps(thing), headers=headers)
            if response.status_code == 201:
                print("Data has been successfully uploaded!")
        except requests.exceptions.RequestException as e:
            print(e)

def video_object_detection_and_georeferencing(video_path, model_path, cls_idx=None, sensor_width_mm=6.4, output="json", start_time=0, end_time=None):
    """
    Performs video object detection and georeferencing, then outputs the data based on the specified method.
    """
    if cls_idx is None:
        cls_idx = [0, 1, 2, 3, 5, 6, 7, 15, 16, 17]  # Default class indexes if not provided

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Used device: {device}")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Set the starting frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Calculate end frame if end_time is provided
    if end_time is not None:
        end_frame = int(end_time * fps)
    else:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_info = parse_srt_file(video_path)
    if frames_info is None:
        return

    frame_count = start_frame

    # List to store data for upload
    Things = []  
     # List to store georeferenced data
    georef_data = [] 
    object_id = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
    output_video_path = '/app/output/processed_video.mp4'
    web_friendly_path = '/app/output/web_friendly_processed_video.mp4'  # Save in the shared volume

    # Delete existing processed videos
    for path in [output_video_path, web_friendly_path]:
        if os.path.exists(path):
            os.remove(path)

    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    
    total_frames = end_frame - start_frame
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= end_frame:
            break

        results = model(frame, verbose=False, classes=cls_idx, device=device)
        result = results[0]

        # Draw bounding boxes on the frame
        for bbox, cls_pred, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = bbox.tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{model.names[int(cls_pred)]}: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
        confidences = np.array(result.boxes.conf.cpu(), dtype=float)
        class_predictions = np.array(result.boxes.cls.cpu(), dtype=int)

        for idx, (cls_pred, bbox, confidence) in enumerate(zip(class_predictions, bboxes, confidences)):
            if frame_count in frames_info:
                frame_data = frames_info[frame_count]
                print(f"Frame {frame_count}: Pitch = {frame_data['gb_pitch']}")

                # Skip object detection and georeferencing if pitch value is above -20 degrees! this may be edited in the future!
                #if frame_data["gb_pitch"] > -20:
                #    continue  # Skip the rest of the loop for this frame

                hfov = 2 * atan2(sensor_width_mm / (2 * frame_data["dzoom_ratio"]), frame_data["focal_length_mm"])
                video_width, video_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                vfov = hfov * (video_height / video_width)

                object_x = (bbox[0] + bbox[2]) / 2  # Midpoint of the width of the bbox; change this if you want to use the center of the bbox
                object_y = bbox[3]  # y coordinate of the bottom edge of the bbox; change this if you want to use the center of the bbox
                delta_x = (object_x - (video_width / 2)) / (video_width / 2)
                delta_y = (object_y - (video_height / 2)) / (video_height / 2)
                    
                angle_to_object_x = delta_x * (degrees(hfov) / 2)
                angle_to_object_y = delta_y * (degrees(vfov) / 2)
                adjusted_bearing = frame_data["gb_yaw"] + angle_to_object_x
                distance_to_object = frame_data["rel_alt"] / sin(radians(-frame_data["gb_pitch"] + angle_to_object_y))

                object_gps_location = calculate_new_gps(frame_data["latitude"], frame_data["longitude"], distance_to_object, adjusted_bearing)
                row = {
                    "klasse": model.names[cls_pred],
                    "frameid": str(frame_count),
                    "lat": str(object_gps_location[0]),
                    "lon": str(object_gps_location[1]),
                }
                
                object_id += 1

                if output == "json":
                    georef_data.append({
                        "class_name": model.names[cls_pred],
                        "coordinates": [object_gps_location[1], object_gps_location[0]],  # [lon, lat]
                        "frameId": frame_count,  # Include frameId
                        "objectId": object_id,  # Includes object ID 
                        "confidence": confidence  # Include confidence 
                    })
                elif output == "server":
                    thing = createThing(row)
                    if thing:
                        Things.append(thing)

        frame_count += 1

        processed_frames += 1
        if processed_frames % 100 == 0:  # Update progress every 100 frames
            print(f"Progress: {processed_frames}/{total_frames} frames processed")

    cap.release()
    out.release()

    if output == "json":
        # Write georeferenced data to JSON file
        output_json_path = '/app/output/data.json'
        with open(output_json_path, "w") as json_file:
            json.dump(georef_data, json_file, indent=4)
        #print("Video processing and georeferencing completed. JSON file has been saved.")
    elif output == "server":
        upload_data_to_server(Things)
        #print("Video processing and georeferencing completed. Data uploaded to server.")
   # Convert video to web-friendly format
    subprocess.run(['ffmpeg', '-i', output_video_path, '-vcodec', 'libx264', '-acodec', 'aac', web_friendly_path])

    # Delete the intermediate processed video
    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    return web_friendly_path  # Return this path instead of output_video_path   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Object Detection and Georeferencing")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("model_path", type=str, help="Path to the YOLO model file")
    parser.add_argument("--cls_idx", nargs='+', type=int, help="List of class indexes to detect")
    parser.add_argument("--output", type=str, default="json", choices=["json", "server"], help="Output destination: 'json' for local file, 'server' for upload")
    args = parser.parse_args()

    # Check if cls_idx was provided, if not, use default values
    if args.cls_idx is None:
        args.cls_idx = [0, 1, 2, 3, 5, 6, 7, 15, 16, 17]

    video_object_detection_and_georeferencing(args.video_path, args.model_path, args.cls_idx, output=args.output)
