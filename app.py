import cv2
import os
from flask import Flask, render_template, Response, request, jsonify
from Llava_model_Inference import run_inference
import re

app = Flask(__name__)

# Define the directory to save captured frames
frames_directory = os.path.join(os.getcwd(), 'static', 'frames')
os.makedirs(frames_directory, exist_ok=True)  # Ensure the directory exists

def gen_frames():
    camera = cv2.VideoCapture(0) 
    while True:
        success, frame = camera.read() 
        frame=cv2.resize(frame,(1000,500),interpolation=cv2.INTER_LINEAR)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])  # Add a new route to handle frame capture
def capture_frame():
    camera = cv2.VideoCapture(0) 
    success, frame = camera.read() 
    frame = cv2.resize(frame, (1000, 500), interpolation=cv2.INTER_LINEAR)

    # Save the captured frame to the specified directory
    filename = os.path.join(frames_directory, 'captured_frame.jpg')
    cv2.imwrite(filename, frame)

    return jsonify({'status': 'success', 'message': 'Frame captured successfully'})

@app.route('/process_prompt', methods=['POST'])
def process_prompt():
    pattern = re.compile(r'ASSISTANT:\s*(.*)\n')

    data = request.get_json()
    query = data.get('prompt', 'Default Prompt')

    # Process the prompt as needed
    print(f'\n\nReceived prompt: {query}')
    answer = run_inference(query)
    match = pattern.search(answer)
    print(match)
    if match:
        # Extract the text using group(1)
        assistant_text = match.group(1).strip()
        print(f'\nAnswer: {assistant_text}\n\n')
    else:
        assistant_text = answer

    return jsonify({'status': 'success', 'processed_prompt': assistant_text})

@app.route('/save_static_image', methods=['POST'])
def save_static_image():

    if 'image' in request.files:
        image_file = request.files['image']
        filename = os.path.join(frames_directory, 'captured_frame.jpg')
        image_file.save(filename)
    
        return jsonify({'status': 'success'})
  
    else:
        return jsonify({'status': 'error'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
