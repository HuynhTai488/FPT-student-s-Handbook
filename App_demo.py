from flask import Flask, Response,jsonify,request, render_template, send_file
import cv2
import PILasOPENCV as Image
import PILasOPENCV as ImageDraw
import PILasOPENCV as ImageFont
from YOLO_Video import video_detection
import base64
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageFont
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Project Detection'

THRESHOLD = 0.3
model=YOLO('D:\Hoc_Tap\Winter_2023\DPL302m\Create_APIs_Flask/best.pt')
classNames = class_name = ['Cây Bàng Đài Loan','Cây Chiều Tím','Cây Dâm Bụt','Cây Hoa Sứ'
                               ,'Cây Huỳnh Anh','Cây Lài Tây', 'Cây Mỏ Két','Cây Ngọc Bút','Cây Nguyệt Quới'
                                ,'Cây Phong Ba','Cây Thùa Lá Hẹp','Cây Trang Son'
                                ,'Cây Tuyết Sơn Phi Hồng','Kí Túc Xá', 'Nhà Tập Võ','Nhà Xe','Sân Banh'
                                ,'Sân Bóng Chuyền - BóngRổ','The Thinker','Trường FPT',]
 

color_by_classnames = [
    (255, 52, 52), (112, 51, 158), (136, 51, 158), (158, 51, 147), (158, 51, 97), (158, 115, 51)
    , (151, 115, 51 ), (133, 158, 51), (112, 158, 51), (72, 158, 51), (51, 158, 76), (51, 158, 97)
    , (51, 158, 126), (51, 151, 158), (51, 97, 158), (51, 72, 158), (158, 83, 51), (158, 51, 79)
    , (154, 51, 158), (90, 51, 158)]


@app.route('/',methods=["POST","GET"])
def home(): 
    return render_template('client.html')

# nginx
@app.route('/upload', methods=["POST"])
def get_output():
    if request.method=='POST':
        image=request.files['image-name']
        image.save("./static/tmp.jpg")
        original_image=cv2.imread("./static/tmp.jpg")

    predictions=model.predict(original_image)
    for index, prediction in enumerate(predictions):
        for object in prediction:
            confidence = object.boxes.conf[0]
            predict_label = int(object.boxes.cls.item())
            print(predict_label, confidence, sep=" ")
            class_color = color_by_classnames[index]
            if confidence > THRESHOLD:
                x1, y1, x2, y2 = [int(point) for point in object.boxes.xyxy.tolist()[0]]
                original_image = cv2.rectangle(original_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                original_image = cv2.putText(original_image, classNames[predict_label], (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, class_color, 2)

    if confidence > THRESHOLD:
        output_image_path = 'static/tmp.jpg'
        cv2.imwrite(output_image_path, original_image)

        response_data = {
            'image_result': output_image_path
        }
        return jsonify(response_data)

if __name__=='__main__':
    app.run(debug=True)