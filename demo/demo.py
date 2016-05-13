import cv2, dlib
from multiprocessing import Process, Queue
import requests
import time
import numpy as np

frame = None
posted_frame = None

TRACK = False
DETECTED = False

URL = 'zhouyz14.deephi.i.deephi.tech:5000/detection/api/'

def req_to_server():
    global frame, posted_frame
    global TRACK
    while True:
        if TRACK:
            time.sleep(0.1)
            pass # do nothing
        else:
            assert frame is not None
            posted_frame = frame # save posted frame
            cv2.imwrite('tmp.jpg', posted_frame)
            f = open('tmp.jpg')
            r = requests.post(url=URL, files={'upload': f})
            resp = r.json()['roi']
            if len(resp) == 0:
                pass
            else:
                DETECTED = True
                dets = np.array(resp)
                print dets
            time.sleep(2)
            pass # post request

def main():
    global frame, posted_frame
    global TRACK

    p = Process(target=f, args=())
    tracker = dlib.correlation_tracker()
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()

    p.start()

    while True:
        ret, frame = cap.read()
        p.put()
        if (TRACK):
            pass

        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()