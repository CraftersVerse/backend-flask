from flask import Flask, request, jsonify , render_template, Response
from PIL import Image
import io
# import os
# import datetime
# import face_recognition
# import time
import math
import cv2
import base64
import numpy as np

from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# from supabase_py import create_client
from dotenv import load_dotenv
load_dotenv()

# url = os.environ.get("SUPABASE_URL")
# key = os.environ.get("SUPABASE_KEY")
# supabase = create_client(url, key)

def perform_object_detection(image_array):
    model=YOLO("./book.pt")
    classNames = ["Book"]

    book_count = 0

    # Perform object detection
    results = model(image_array)

    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(image_array, (x1,y1), (x2,y2), (255,0,255),3)
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            class_name=classNames[cls]

            if class_name == "Book":
                book_count += 1

            label=f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(image_array, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
            cv2.putText(image_array, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
    return image_array, book_count

def image_to_base64(image):
    # Convert the OpenCV image array to a PIL Image
    pil_image = Image.fromarray(image)

    # Convert the PIL Image to JPEG format
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")

    # Encode the image in Base64
    base64_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Build the data URI string
    data_uri = f'data:image/jpeg;base64,{base64_encoded}'

    return data_uri

@app.route('/predict_objects', methods=['POST'])
def predict_objects():
    # print(request.files.getlist('images_0'))
    if 'images_0' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    predicted_images_data = []
    
    # print(len(request.files))
    for index in range(len(request.files)):
        key = f'images_{index}'
        # print(f'images_{index}')
        # uploaded_images = request.files.getlist(f'image_{index}')
        uploaded_images = request.files.getlist(key)
        print('this is image data:',uploaded_images)

        for uploaded_image in uploaded_images:
            # Read the image using OpenCV
            image_array = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

            # Perform object detection
            detected_image, book_count = perform_object_detection(image_array)

            # Convert the detected image to Base64-encoded data URI
            data_uri = image_to_base64(detected_image)
            predicted_images_data.append({'data_uri': data_uri, 'book_count': book_count})
            # print(predicted_images_base64_data_url)

    return jsonify({'predicted_images_data': predicted_images_data}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
