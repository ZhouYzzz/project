import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.putText(frame, 'HAHAH', (200,200), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 2)
    cv2.imshow('cam', frame)

    cv2.waitKey(1)


