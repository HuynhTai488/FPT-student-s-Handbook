from flask import Flask, Response,jsonify,request, render_template
import cv2
from YOLO_Video import video_detection
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Project Detection'



def generate_frames(path_x = ''):

    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg',detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/',methods=['POST',"GET"])
def home():
    return render_template('client.html')

@app.route('/video')
def video():
    return Response(generate_frames(path_x=r"C:\Users\htai8\Downloads\VID_20231016_164149 (1).mp4"), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)