from flask import Flask, Response,jsonify,request, render_template, send_file, redirect, url_for
import cv2


import base64
import tensorflow as tf
from PIL import Image,ImageDraw, ImageFont

from werkzeug.utils import secure_filename
import numpy as np
from ultralytics import YOLO
from PIL import Image

import os
from ultralytics import YOLO
import cv2
import math

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Project Detection'

model=YOLO('./best.pt')
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
#CREATE THRESHOLD
THRESHOLD=0.3

# Định nghĩa hàm class_name để ánh xạ từ mã số lớp sang label tiếng Việt
def class_name(predict_label):
    return class_names[predict_label]

from flask import Response

def generate_frames(path_x = ''):

    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg',detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # print(x1, y1, x2, y2)
                predict_label = int(box.cls.item())
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                # cv2.putText(img, label, (x1, y1-2),0, 1,[255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                # Chuyển predict_label thành label tiếng Việt
                label_vi = class_name

                # Sử dụng Pillow để vẽ bounding box với label tiếng Việt
                pil_image = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_image)
                font_path = ".\\Font_Unicode\\Roboto-Regular.ttf"
                font = ImageFont.truetype(font_path, 50)
                draw.text((x1, y1 - 80), label_vi, fill="lightblue", font=font)
                original_image = np.array(pil_image)
        yield original_image
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()


@app.route('/',methods=['POST',"GET"])
def home():
    return render_template('client.html')

@app.route('/upload', methods=["POST"])
def get_output():
    if request.method == 'POST':
        image = request.files['image-name']
        image.save("./static/tmp.jpg")
        original_image = cv2.imread("./static/tmp.jpg")

    predictions = model.predict(original_image)
    for index, prediction in enumerate(predictions):
        for object in prediction:
            confidence = object.boxes.conf[0]
            predict_label = int(object.boxes.cls.item())
            print(predict_label, confidence, sep=" ")
            class_color = (0, 0, 255)  # Màu đỏ (RGB)

            if confidence > THRESHOLD:
                x1, y1, x2, y2 = [int(point) for point in object.boxes.xyxy.tolist()[0]]
                original_image = cv2.rectangle(original_image, (x1, y1), (x2, y2), color=class_color, thickness=2, lineType=cv2.LINE_AA)

                # Chuyển predict_label thành label tiếng Việt
                label_vi = class_name(predict_label)

                # Sử dụng Pillow để vẽ bounding box với label tiếng Việt
                pil_image = Image.fromarray(original_image)
                draw = ImageDraw.Draw(pil_image)
                font_path = ".\\Font_Unicode\\Roboto-Regular.ttf"
                font = ImageFont.truetype(font_path, 30)
                draw.text((x1, y1 - 30), label_vi, fill="lightblue", font=font)
                original_image = np.array(pil_image)

    if confidence > THRESHOLD:
        output_image_path = 'static/tmp.jpg'
        cv2.imwrite(output_image_path, original_image)
        
        response_data = {
            'image_result': output_image_path
        }
        return jsonify(response_data)
    else:
        response_data = {
            'image_result': None
        }
        return jsonify(response_data)
# Khởi tạo biến toàn cục để lưu đường dẫn video
video_path = None

@app.route('/video', methods=['POST',"GET"])
def upload_video():
    if request.method == 'POST':
        video = request.files['video']
        if video:
            video.save("./static/video.mp4")
            video_path = cv2.imread("./static/video.mp4")  
    #         return redirect(url_for('view_video'))
    # return render_template('client_video.html')
            return Response(generate_frames('./static/video.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('client_video.html')
#Xem video đã lưu
@app.route('/view_video')
def view_video():
    video_path='/static/video.mp4'  #đường dẫ tơi video đã lưu
    return render_template('view_video.html',video_path=video_path)


#Webcam
@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)