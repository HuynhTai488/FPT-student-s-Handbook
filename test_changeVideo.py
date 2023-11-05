from flask import Flask, Response,jsonify,request, render_template, send_file, redirect, url_for
import cv2
import os

import base64
import tensorflow as tf
from PIL import Image,ImageDraw, ImageFont


import numpy as np
from ultralytics import YOLO
from PIL import Image

from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import math

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Project Detection'

model=YOLO('D:\Hoc_Tap\Winter_2023\DPL302m\Create_APIs_Flask/best.pt')
classNames = class_names = [''] + ['Cây Bàng Đài Loan','Cây Chiều Tím','Cây Dâm Bụt','Cây Hoa Sứ'
                               ,'Cây Huỳnh Anh','Cây Lài Tây', 'Cây Mỏ Két','Cây Ngọc Bút','Cây Nguyệt Quới'
                                ,'Cây Phong Ba','Cây Thùa Lá Hẹp','Cây Trang Son'
                                ,'Cây Tuyết Sơn Phi Hồng','Kí Túc Xá', 'Nhà Tập Võ','Nhà Xe','Sân Banh'
                                ,'Sân Bóng Chuyền - BóngRổ','The Thinker','Trường FPT',]

color_by_classnames = [
    (255, 52, 52), (112, 51, 158), (136, 51, 158), (158, 51, 147), (158, 51, 97), (158, 115, 51)
    , (151, 115, 51 ), (133, 158, 51), (112, 158, 51), (72, 158, 51), (51, 158, 76), (51, 158, 97)
    , (51, 158, 126), (51, 151, 158), (51, 97, 158), (51, 72, 158), (158, 83, 51), (158, 51, 79)
    , (154, 51, 158), (90, 51, 158)]
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response, request

THRESHOLD = 0.3

# Định nghĩa hàm class_name để ánh xạ từ mã số lớp sang label tiếng Việt
def class_name(predict_label):
    return class_names[predict_label]

def video_detection(video_data):
    cap = cv2.VideoCapture(video_data)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    original_image = None  # Khởi tạo original_image ở đây

    while True:
        success, img = cap.read()
        if not success:
            break

        if original_image is None:
            original_image = img.copy()  # Sao chép img để khởi tạo original_image

        # Thực hiện dự đoán của mô hình YOLO
        predictions = model.predict(img)
        #lặp qua các dự đoán trong video vì 1 video có nhiều đối tượng
        for index, prediction in enumerate(predictions):
            for object in prediction:
                # Đối tượng (object) có một thuộc tính boxes chứa các  (bounding boxes) và thông tin khác. 
                # Thuộc tính conf cho biết độ tự tin (confidence) của mô hình về việc dự đoán đối tượng
                confidence = object.boxes.conf[0]
                predict_label = int(object.boxes.cls.item())
                print(predict_label, confidence, sep=" ")
                class_color = (0, 0, 255)  # Màu đỏ (RGB)

                if confidence > THRESHOLD:
                    #Lấy các điểm x,y,x',y' trong 
                    x1, y1, x2, y2 = [int(point) for point in object.boxes.xyxy.tolist()[0]]

                    label_vi = class_name(predict_label)
                    original_image = cv2.rectangle(original_image, (x1, y1), (x2, y2), color=class_color, thickness=2, lineType=cv2.LINE_AA)
                    #Bước chuyển font chữ roboto để hỗ trợ tiếng việt

                    pil_image = Image.fromarray(original_image)
                    draw = ImageDraw.Draw(pil_image)
                    font_path = "D:\\Hoc_Tap\\Winter_2023\\DPL302m\\Create_APIs_Flask\\Font_Unicode\\Roboto-Regular.ttf"
                    font = ImageFont.truetype(font_path, 60)
                    draw.text((x1, y1 - 60), label_vi, fill='lightblue', font=font)
                    original_image = np.array(pil_image)
        yield original_image

    cap.release()
    # cv2.destroyWindow()
def generate_frames(video_data):
    yolo_output = video_detection(video_data)
    for frame in yolo_output:
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Đảm bảo thư mục uploads tồn tại
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET'])
def home_video():
    return render_template('client_video.html')

@app.route('/video', methods=['POST'])
def upload_video():
    #Check file có chưa
    if 'video' not in request.files:
        return redirect(url_for('home_video'))

    video_file = request.files['video']
    if video_file.filename == '':
        return redirect(url_for('home_video'))

    if video_file:
        #Dùng hàm secure_file trong module werkzeug.utils để bảo vệ an toàn cho file
        filename = secure_filename(video_file.filename)
        video_path = os.path.join('static', filename)  # Lưu vào thư mục "static" với đuôi là mp4
        video_file.save(video_path)

        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)