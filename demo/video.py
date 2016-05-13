import cv2

cap = cv2.VideoCapture(1)

while(True):
    ret, frame = cap.read()
    cv2.imshow('cam', frame)
    cv2.waitKey(1)

print frame.shape
