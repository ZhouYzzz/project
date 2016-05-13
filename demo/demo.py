import cv2, dlib
from threading import Thread
import requests
import time
import numpy as np

frame = None
posted_frame = None
person_im = None

TRACK = False
DETECTED = False
END = False

URL_detect = 'http://zhouyz14.deephi.i.deephi.tech:5000/detection/api/'
URL_reid = 'http://zhouyz14.deephi.i.deephi.tech:5000/reid/req_to_database/'
rect = None
detected_rect = None

def req_to_server():
    global frame, posted_frame
    global TRACK, DETECTED, END
    global detected_rect
    while True:
        # print frame[0,0,0]
        if END:
            exit()
        if TRACK:
            time.sleep(0.1)
            pass # do nothing
        else:
            assert frame is not None
            posted_frame = frame # save posted frame
            cv2.imwrite('tmp.jpg', posted_frame)
            # print posted_frame[0,0,0]
            f = open('tmp.jpg')
            r = requests.post(url=URL_detect, files={'upload': f})
            resp = r.json()['roi']
            print resp
            if len(resp) == 0:
                pass
            else:
                DETECTED = True
                TRACK = True
                print 'Detected!!!'
                dets = np.array(resp)
                print dets
                detected_rect = dets[0,:4].astype(int).tolist()
            time.sleep(4)
            pass # post request

def req_reid():
    global person_im, rect
    while True:
        if rect is not None:
            print rect
            person_im = frame[rect[1]:rect[3],rect[0]:rect[2],:]

            cv2.imwrite('person_im.jpg', person_im)
            f = open('person_im.jpg')
            r = requests.post(url=URL_reid, files={'upload': f})
            print r.text
            time.sleep(1)
        else:
            time.sleep(0.5)

def main():
    global frame, posted_frame
    global TRACK, DETECTED, END
    global rect, detected_rect

    tracker = dlib.correlation_tracker()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    trd_post = Thread(target=req_to_server)
    trd_post.start()

    trd_reid = Thread(target=req_reid)
    trd_reid.start()

    while True:
        ret, frame = cap.read()
        # print frame[0,0,0]
        if TRACK:
            if DETECTED:
                tracker.start_track(posted_frame,
                        dlib.rectangle(*detected_rect))
                tracker.update(frame)
                DETECTED = False
            else:
                tracker.update(frame)
        
            bbox = tracker.get_position()
            pt1 = (int(bbox.left()), int(bbox.top()))
            pt2 = (int(bbox.right()), int(bbox.bottom()))
            rect = pt1 + pt2
            # print pt1, pt2, rect
            cv2.rectangle(frame, pt1, pt2, (255,255,255), 2)

        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    END = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
