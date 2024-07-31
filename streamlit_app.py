import streamlit as st
import requests
import os
import atexit
import time
import folium
from streamlit_folium import folium_static
import json
import cv2
from PIL import Image
import io
import numpy as np
import base64
from urllib.parse import urlencode, parse_qs

st.set_page_config(layout="wide", page_title="VIGODT", page_icon="🌍")
st.title('Visual Interface for Georeferencing and Object Detection and Tracking - VIGODT')

# Function to check if corresponding SRT file exists
def check_srt_file(video_file):
    srt_file = os.path.splitext(video_file)[0] + '.srt'
    srt_file_upper = os.path.splitext(video_file)[0] + '.SRT'
    return os.path.exists(os.path.join(st.session_state.current_dir, srt_file)) or \
        os.path.exists(os.path.join(st.session_state.current_dir, srt_file_upper))

def read_url_params():
    query_params = st.query_params
    if 'dir' in query_params:
        st.session_state.current_dir = query_params['dir']
    if 'video' in query_params:
        video_path = os.path.join(st.session_state.current_dir, query_params['video'])
        if os.path.exists(video_path):
            st.session_state.selected_video = query_params['video']
        else:
            st.session_state.selected_video = None
    if 'start' in query_params:
        st.session_state.start_time = float(query_params['start'])
    if 'end' in query_params:
        st.session_state.end_time = float(query_params['end']) if query_params['end'] != 'None' else None
    if 'action' in query_params:
        if query_params['action'] == 'process':
            st.session_state.processing = True
        elif query_params['action'] == 'show_map':
            st.session_state.processing_complete = True

def update_url_params():
    if st.session_state.selected_video:
        params = {
            'dir': st.session_state.current_dir,
            'video': st.session_state.selected_video,
            'start': st.session_state.start_time,
            'end': st.session_state.end_time if st.session_state.end_time is not None else 'None',
            'action': 'show_map' if st.session_state.processing_complete else 'process'
        }
    else:
        params = {'dir': st.session_state.current_dir}
    
    # Clear all existing parameters
    for key in list(st.query_params.keys()):
        del st.query_params[key]
    
    # Update with new parameters
    st.query_params.update(params)

# Custom CSS to change button color, behavior, and slider color
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #009878;
    color: white;
    border: none;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background-color: #00876a;
    color: white !important;
    transform: scale(1.05);
}

div.stButton > button:active, div.stButton > button:focus {
    background-color: #00876a;
    color: white !important;
    transform: scale(0.98);
    border: 2px solid #3498db;
    outline: none;
    box-shadow: none;
}

div.stButton > button:focus:not(:active) {
    border: 2px solid #3498db;
    transform: scale(1.05);
}
            
.video-thumbnail {
    border: 3px solid transparent;
    padding: 10px;
    margin-bottom: 20px;
}

.video-thumbnail.selected {
    border-color: #00FF00;
}

.video-title {
    font-weight: bold;
    margin-bottom: 10px;
}

.select-button {
    background-color: #009878;
    color: white;
    border: none;
    padding: 5px 10px;
    cursor: pointer;
}


</style>
""", unsafe_allow_html=True)

# Define the base directory
base_dir = '/data'

# Initialize session state
if 'current_dir' not in st.session_state:
    st.session_state.current_dir = base_dir
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0
if 'end_time' not in st.session_state:
    st.session_state.end_time = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Read URL parameters
read_url_params()
read_url_params()

    # Right after reading URL parameters
if st.session_state.selected_video:
    st.session_state.srt_exists = check_srt_file(st.session_state.selected_video)

#st.write("Debug: Session state after checking SRT:", st.session_state)

# Clear processing-related parameters if no video is selected
if not st.session_state.selected_video:
    st.session_state.processing = False
    st.session_state.processing_complete = False
    if 'action' in st.query_params:
        del st.query_params['action']
    update_url_params()

    # Add this after initializing session state
if 'action' in st.query_params:
    if st.query_params['action'] == 'process':
        st.session_state.processing = True
        st.session_state.processing_complete = False
    elif st.query_params['action'] == 'show_map':
        st.session_state.processing_complete = True

# Read URL parameters
read_url_params()

# Define custom icons
icons = {
    'car': 'https://svgsilh.com/svg/309541.svg',
    'person': 'https://svgsilh.com/svg/297255.svg',
    'truck': 'https://svgsilh.com/svg/1918551.svg',
    'container': 'https://svgsilh.com/svg/1576079.svg',
    'motorcycle': 'https://svgsilh.com/svg/1131863.svg',
    'bus': 'https://svgsilh.com/svg/296715.svg',
    'train': 'https://www.svgrepo.com/show/115517/train.svg'
}

# Function to create map
def create_map(data, start_frame, end_frame, confidence_threshold):
    # Filter data for the selected frame range and confidence threshold
    filtered_data = [obj for obj in data if start_frame <= obj['frameId'] <= end_frame and obj['confidence'] >= confidence_threshold]
    
    if not filtered_data:
        # If no data in range, use a default location (you can change this)
        return folium.Map(location=[0, 0], zoom_start=2)
    
    # Calculate average lat and lon
    lats = [obj['coordinates'][1] for obj in filtered_data if obj['coordinates']]
    lons = [obj['coordinates'][0] for obj in filtered_data if obj['coordinates']]
    
    if not lats or not lons:
        # If no valid coordinates, use a default location
        return folium.Map(location=[0, 0], zoom_start=2)
    
    avg_lat = sum(lats) / len(lats)
    avg_lon = sum(lons) / len(lons)
    
    # Calculate appropriate zoom level based on coordinate spread
    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)
    max_range = max(lat_range, lon_range)
    
    if max_range < 0.01:
        zoom_start = 15
    elif max_range < 0.1:
        zoom_start = 13
    elif max_range < 1:
        zoom_start = 10
    else:
        zoom_start = 8
    
    # Create the map centered on the average coordinates
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom_start)
    
    # Add markers as before
    for obj in filtered_data:
        coordinates = obj['coordinates']
        if coordinates:
            class_name = obj['class_name']
            frame_id = obj['frameId']
            object_id = obj['objectId']
            confidence = obj['confidence']

            popup_text = f"Class: {class_name}<br>FrameID: {frame_id}<br>ObjectID: {object_id}<br>Confidence: {confidence:.2f}"
            
            icon_url = icons.get(class_name.lower(), 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png')
            
            icon = folium.CustomIcon(
                icon_image=icon_url,
                icon_size=(38, 95),
                icon_anchor=(19, 95),
                popup_anchor=(0, -95)
            )
            
            folium.Marker(
                location=[coordinates[1], coordinates[0]],
                popup=popup_text,
                tooltip=class_name,
                icon=icon
            ).add_to(m)
    
    return m

# Create three columns for layout
left_column, middle_column, right_column = st.columns([30, 35, 35])

# Function to delete processed video
def delete_processed_video(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted processed video: {path}")

# Register cleanup function to run on exit
atexit.register(delete_processed_video, '/app/output/web_friendly_processed_video.mp4')

@st.cache_data
def list_files_and_dirs(current_dir):
    try:
        dirs = []
        mp4_files = []
        with os.scandir(current_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    dirs.append(entry.name)
                elif entry.is_file() and entry.name.upper().endswith('.MP4'):
                    mp4_files.append(entry.name)
        return dirs, mp4_files
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return [], []

@st.cache_data
def generate_thumbnail(video_path, max_size=(400, 300)):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        aspect = w / h
        if w > h:
            new_w = min(w, max_size[0])
            new_h = int(new_w / aspect)
        else:
            new_h = min(h, max_size[1])
            new_w = int(new_h * aspect)
        thumbnail = cv2.resize(frame, (new_w, new_h))
        return Image.fromarray(np.uint8(thumbnail))
    return None

# Session state initialization
if 'current_dir' not in st.session_state:
    st.session_state.current_dir = base_dir
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = None
if 'srt_exists' not in st.session_state:
    st.session_state.srt_exists = False
if 'current_processed_video' not in st.session_state:
    st.session_state.current_processed_video = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

    # Read URL parameters
if 'dir' in st.query_params:
    st.session_state.current_dir = st.query_params['dir']
if 'video' in st.query_params:
    st.session_state.selected_video = st.query_params['video']
if 'start' in st.query_params:
    st.session_state.start_time = float(st.query_params['start'])
if 'end' in st.query_params:
    end_param = st.query_params['end']
    st.session_state.end_time = float(end_param) if end_param != 'None' else None
else:
    st.session_state.end_time = None
if 'action' in st.query_params and st.query_params['action'] == 'process':
    st.session_state.processing = True



with left_column:
    # Display current directory
    st.write(f"Current Directory: {st.session_state.current_dir}")

    if st.button('Go Up One Directory 📂', key="go_up_button"):
        parent_dir = os.path.dirname(st.session_state.current_dir)
        if os.path.commonpath([base_dir, parent_dir]) == base_dir:
            st.session_state.current_dir = parent_dir
            st.session_state.selected_video = None
            st.session_state.srt_exists = False
            st.session_state.processing_complete = False
            st.session_state.processing = False
            update_url_params()
            st.rerun()
        else:
            st.warning("You are already at the root directory.")

    # Display selected video at the top
    if st.session_state.selected_video:
        st.write(f"**Selected Video: {st.session_state.selected_video}**")
        if st.session_state.srt_exists:
            st.write("SRT file found")
        else:
            st.write("No SRT file found")
        
        if st.button('Back to Directory', key="back_to_directory"):
            st.session_state.selected_video = None
            st.session_state.srt_exists = False
            st.session_state.processing_complete = False
            update_url_params()
            st.rerun()
        
        st.markdown("---")

    # List directories and files
    dirs, mp4_files = list_files_and_dirs(st.session_state.current_dir)

    # Display directories with navigation buttons only if there are subdirectories
    if dirs:
        st.write("Subdirectories:")
        for dir_name in dirs:
            if st.button(f"📁 {dir_name}", key=f"dir_{dir_name}"):
                st.session_state.current_dir = os.path.join(st.session_state.current_dir, dir_name)
                st.session_state.selected_video = None
                st.session_state.srt_exists = False
                st.session_state.processing_complete = False
                st.session_state.processing = False
                update_url_params()
                st.rerun()
        st.markdown("---")

    # Pagination for thumbnails
    items_per_page = 10
    total_pages = (len(mp4_files) - 1) // items_per_page + 1
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1

    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    # Display thumbnails with pagination
    st.write("Video Thumbnails:")
    for mp4_file in mp4_files[start_idx:end_idx]:
        video_path = os.path.join(st.session_state.current_dir, mp4_file)
        
        # Create HTML for the thumbnail
        thumbnail = generate_thumbnail(video_path, max_size=(400, 300))
        if thumbnail:
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            thumbnail.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            html = f"""
            <div class="video-thumbnail {'selected' if mp4_file == st.session_state.selected_video else ''}">
                <div class="video-title">{mp4_file}</div>
                <img src="data:image/png;base64,{img_str}" style="width:100%">
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.write(f"**{mp4_file}** - No thumbnail available")
        
        # Add the select button outside of the custom HTML
        if st.button("Select", key=f"select_{mp4_file}"):
            st.session_state.selected_video = mp4_file
            st.session_state.srt_exists = check_srt_file(mp4_file)
            update_url_params()
            st.rerun()
        
        st.markdown("---")

    # Pagination controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous Page", key="prev_page_button") and st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()
    with col2:
        st.write(f"Page {st.session_state.current_page} of {total_pages}")
    with col3:
        if st.button("Next Page", key="next_page_button") and st.session_state.current_page < total_pages:
            st.session_state.current_page += 1
            st.rerun()

    # Display MP4 files found
    st.write(f"MP4 files found: {len(mp4_files)}")

    # Display "View on Map!" button if processing is complete
    if st.session_state.processing_complete:
        if st.button('View Map in a new Tab!', key="view_map_button"):
            js = f'''
                <script>
                    window.open('http://localhost:5000', '_blank').focus();
                </script>
                '''
            st.components.v1.html(js)

with middle_column:
    if st.session_state.selected_video:
        st.write(f"**Selected Video: {st.session_state.selected_video}**")
        
        # Check for SRT file
        st.session_state.srt_exists = check_srt_file(st.session_state.selected_video)
        
        if st.session_state.srt_exists:
            st.write("SRT file found")
            
            # Get video duration
            video_path = os.path.join(st.session_state.current_dir, st.session_state.selected_video)
            cap = cv2.VideoCapture(video_path)
            duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            # Time range selection
            st.write("Select time range for processing:")
            start_time = st.slider(
                "Start time (seconds)", 
                0, 
                duration, 
                int(st.session_state.start_time), 
                key=f"start_time_slider_{st.session_state.selected_video}"
            )
            end_time = st.slider(
                "End time (seconds)", 
                start_time, 
                duration, 
                int(st.session_state.end_time or duration), 
                key=f"end_time_slider_{st.session_state.selected_video}"
            )

            # Process video button
            process_button = st.button('Process Video!', key=f"process_video_button_{st.session_state.selected_video}")

            # Check if we should process the video
            should_process = process_button or (st.session_state.processing and 'action' in st.query_params and st.query_params['action'] == 'process')

            if should_process:
                st.session_state.processing = True
                st.session_state.processing_complete = False
                
                # Call the video processing API
                try:
                    with st.spinner('Processing video... This may take a while for large files.'):
                        response = requests.post('http://video_processing:5001/process_video', 
                                                json={
                                                    'video_path': video_path,
                                                    'model_path': 'yolov8n.pt',
                                                    'start_time': start_time,
                                                    'end_time': end_time
                                                })
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success('Processing completed!')
                        
                        # Store the new processed video path
                        st.session_state.current_processed_video = result['output_video_path']
                        st.session_state.processing_complete = True
                        
                        # Update URL to indicate processing is complete
                        st.query_params['action'] = 'show_map'
                        update_url_params()
                    else:
                        st.error(f"Error processing video: {response.text}")
                except requests.RequestException as e:
                    st.error(f"Error calling video processing service: {str(e)}")
                
                st.session_state.processing = False
                st.rerun()

            # Display processed video if available
            if st.session_state.processing_complete and st.session_state.current_processed_video:
                st.video(st.session_state.current_processed_video)

        else:
            st.error("No SRT file found for the selected video. Processing cannot continue.")
    else:
        st.write("No video selected. Please select a video from the left column.")

    # Debug information
    #st.write("Debug: Current session state:", st.session_state)
    #st.write("Debug: Current query parameters:", st.query_params)

with right_column:
    if st.session_state.processing_complete or st.query_params.get('action') == 'show_map':
        try:
            with open('/app/output/data.json', 'r') as f:
                data = json.load(f)
            
            if data:
                # Get the range of frame IDs
                frame_ids = [obj['frameId'] for obj in data]
                min_frame = min(frame_ids)
                max_frame = max(frame_ids)

                # Initialize session state for selected frame and show all frames
                if 'selected_frame' not in st.session_state:
                    st.session_state.selected_frame = min_frame
                if 'show_all_frames' not in st.session_state:
                    st.session_state.show_all_frames = True
                if 'confidence_threshold' not in st.session_state:
                    st.session_state.confidence_threshold = 0.0

                # Add confidence threshold slider
                new_confidence_threshold = st.slider(
                    "Confidence threshold:",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.confidence_threshold,
                    step=0.05
                )

                # Create the map
                if st.session_state.show_all_frames:
                    m = create_map(data, min_frame, max_frame, new_confidence_threshold)
                else:
                    m = create_map(data, st.session_state.selected_frame, st.session_state.selected_frame, new_confidence_threshold)
            
                # Use custom CSS to make the map responsive
                st.markdown("""
                <style>
                .fullScreenFrame > div {
                    width: 100%;
                    height: 100%;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display the map
                with st.container():
                    folium_static(m, width=1000, height=700)

                # Add controls under the map
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Add a single slider for frame selection
                    new_selected_frame = st.slider(
                        "Select frame:",
                        min_value=min_frame,
                        max_value=max_frame,
                        value=st.session_state.selected_frame
                    )

                with col2:
                    # Add a checkbox to select all frames
                    new_show_all_frames = st.checkbox("Show all frames", value=st.session_state.show_all_frames)

                # Check if any control has changed
                if (new_selected_frame != st.session_state.selected_frame or 
                    new_show_all_frames != st.session_state.show_all_frames or 
                    new_confidence_threshold != st.session_state.confidence_threshold):
                    st.session_state.selected_frame = new_selected_frame
                    st.session_state.show_all_frames = new_show_all_frames
                    st.session_state.confidence_threshold = new_confidence_threshold
                    st.rerun()
            else:
                st.warning("No data available to display on the map.")
        except FileNotFoundError:
            st.error("Data file not found. Make sure the processing has completed successfully.")
        except json.JSONDecodeError:
            st.error("Error reading the data file. The file might be empty or corrupted.")
    else:
        st.write("Process a video to see the map and data visualization.")