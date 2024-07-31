from flask import Flask, request, jsonify
from detectiongeoref_final import video_object_detection_and_georeferencing

app = Flask(__name__)

@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    video_path = data.get('video_path')
    model_path = data.get('model_path')
    start_time = data.get('start_time', 0)
    end_time = data.get('end_time')
    
    if not video_path or not model_path:
        return jsonify({'status': 'error', 'message': 'Missing video_path or model_path'}), 400
    
    try:
        output_video_path = video_object_detection_and_georeferencing(
            video_path, model_path, output="json", start_time=start_time, end_time=end_time
        )
        return jsonify({'status': 'success', 'output_video_path': output_video_path}), 200
    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)