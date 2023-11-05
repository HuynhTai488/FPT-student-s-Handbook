import cv2
from flask import Flask, render_template,Response
import math
from ultralytics import YOLO
app = Flask(__name__)

def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

    model=YOLO('D:\Hoc_Tap\Winter_2023\DPL302m\Create_APIs_Flask\yolov8n.pt')
    classNames = class_name = ['Cây Bàng Đài Loan','Cây Chiều Tím','Cây Dâm Bụt','Cây Hoa Sứ'
                               ,'Cây Huỳnh Anh','Cây Lài Tây', 'Cây Mỏ Két','Cây Ngọc Bút','Cây Nguyệt Quới'
                                ,'Cây Phong Ba','Cây Thùa Lá Hẹp','Cây Trang Son'
                                ,'Cây Tuyết Sơn Phi Hồng','Kí Túc Xá', 'Nhà Tập Võ','Nhà Xe','Sân Banh'
                                ,'Sân Bóng Chuyền - BóngRổ','The Thinker','Trường FPT',]
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # print(x1, y1, x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1-2),0, 1,[255, 255, 255], thickness=1,lineType=cv2.LINE_AA)
        yield img
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()

# Hàm để đọc dữ liệu từ webcam và gửi dữ liệu frame tới client real-time
def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for frame in yolo_output:
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
# Route cho trang chứa video từ webcam
@app.route('/', methods=['POST',"GET"])
def home():
    return render_template('client.html')
         
@app.route('/webcam')
def webcam():
    return render_template('client_webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)   
