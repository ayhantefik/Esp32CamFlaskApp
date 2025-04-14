from io import BytesIO
from PIL import Image
from flask import Flask, Response, render_template, request, jsonify
from base64 import b64encode
import pytesseract
from flask_socketio import SocketIO
import fingercount_model
import fruits_model
import cv2
import numpy as np
import time

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secretkey!'
socketio = SocketIO(app)

result = ""
page_value = 0 #1 = text recognition, 2 = finger counting, 3 = fruit detection

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def index():
    return Response(get_image(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_image():
    global result
    global page_value
    
    last_result = None
    prev_image_bytes = None
    
    # Record the start time
    start_time = time.time()
    # Initialize the last printed time
    last_print_time = start_time
    # Interval in seconds
    print_interval = 2.5
    while True:
        current_time = time.time()
        try:
            with open("image.jpg", "rb") as f:
                image_bytes = f.read()
            image = Image.open(BytesIO(image_bytes))
            if current_time - last_print_time >= print_interval:
                # Update the last printed time
                last_print_time = current_time
                # Calculate total elapsed time from start
                total_elapsed_time = current_time - start_time
                if prev_image_bytes is not None:
                    if(is_change_detected(prev_image_bytes, image_bytes)):
                        if(page_value == 1):
                            read_frame = read_frame_from_bytes(image_bytes)
                            frame_gray = cv2.cvtColor(read_frame, cv2.COLOR_BGR2GRAY)
                            # Apply Gaussian Blur
                            blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
                            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
                            kernel = np.ones((2,2), np.uint8)
                            dilation = cv2.dilate(thresh, kernel, iterations=0)
                            result = pytesseract.image_to_string(dilation)
                        elif (page_value == 2):
                            result = fingercount_model.get_model_result()
                        elif (page_value == 3):
                            result = fruits_model.get_model_result()
                if (result != last_result):
                    socketio.emit('message', result)
                    last_result = result
                prev_image_bytes = image_bytes
                
            img_io = BytesIO()
            image.save(img_io, 'JPEG')
            img_io.seek(0)
            img_bytes = img_io.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

        except Exception as e:
            print("encountered an exception: ")
            print(e)

            with open("placeholder.jpg", "rb") as f:
                image_bytes = f.read()
            image = Image.open(BytesIO(image_bytes))
            img_io = BytesIO()
            image.save(img_io, 'JPEG')
            img_io.seek(0)
            img_bytes = img_io.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
            continue
        
        # Sleep for a short time to prevent the loop from running too fast
        time.sleep(0.1)

def read_frame_from_bytes(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def is_change_detected(prev_image_bytes, current_image_bytes):
    global page_value
    read_prev_frame = read_frame_from_bytes(prev_image_bytes)
    read_current_frame = read_frame_from_bytes(current_image_bytes)
    
    # Convert images to grayscale
    prev_frame_gray = cv2.cvtColor(read_prev_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(read_current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the difference and threshold
    frame_diff = cv2.absdiff(prev_frame_gray, current_frame_gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Analyze changes
    non_zero_count = np.count_nonzero(thresh)
    if(page_value == 1):
        if non_zero_count > 2000:
            return True
        else:
            return False
    else:
        if non_zero_count > 5000:
            return True
        else:
            return False

@app.route('/switch_value', methods=['POST'])
def switch_value():
    global page_value
    data_received = request.json
    page_value = int(data_received['value'])
    print(page_value)
    # Process the data as needed
    return jsonify({'status': 'success', 'received_data': data_received})

@socketio.on('message')
def handlemsg(msg):
    global result
    socketio.send(result)

if __name__ == '__main__':
    socketio.run(app)
