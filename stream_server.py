import threading, time, cv2
from flask import Flask, Response

app = Flask(__name__)
frame_queue = None        # 외부에서 할당
latest_frames = {}        # stream_name: frame

def run_stream_server(queue, host='0.0.0.0', port=5000):
    global frame_queue
    frame_queue = queue

    def queue_consumer():
        while True:
            try:
                name, frame = frame_queue.get()
                latest_frames[name] = frame
            except Exception as e:
                print("Queue error:", e)

    threading.Thread(target=queue_consumer, daemon=True).start()

    def flask_thread():
        print(f"[INFO] Flask MJPEG server running: http://{host}:{port}")
        app.run(host=host, port=port, threaded=True, debug=False)
    threading.Thread(target=flask_thread, daemon=True).start()

@app.route('/')
def index():
    items = ''.join(f'<li><a href="/{n}" target="_blank">{n}</a></li>' for n in latest_frames.keys())
    return f'<html><body><h1>Available Streams</h1><ul>{items}</ul></body></html>'

@app.route('/<stream_name>')
def mjpeg(stream_name):
    def gen():
        boundary = b'--frame'
        while True:
            frame = latest_frames.get(stream_name)
            if frame is not None:
                ret, jpg = cv2.imencode('.jpg', frame)
                if ret:
                    yield boundary + b'\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')





# 아래는 (메인 프로세스에서) 서버 띄우는 예시일 뿐, 외부에서 큐만 공유하면 됨
if __name__ == "__main__":
    from multiprocessing import Queue
    q = Queue()
    run_stream_server(q)    # 여기서 서버 시작

    # 아래는 테스트: 임의의 프레임 계속 넣기
    import numpy as np
    while True:
        img_1 = (np.random.rand(240, 320, 3) * 255).astype('uint8')
        q.put(('test_1', img_1))
        img_2 = (np.random.rand(240, 320, 3) * 255).astype('uint8')
        q.put(('test_2', img_2))
        time.sleep(1/30)
