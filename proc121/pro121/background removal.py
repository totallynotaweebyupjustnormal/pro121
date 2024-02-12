import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3, 640)
camera.set(4, 480)

rmountain = cv2.imread('mount_everest.jpg')

mountain = cv2.resize(mountain, (640, 480))

while True:
    status, frame = camera.read()

    if status:
        frame = cv2.flip(frame, 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # creating thresholds (example: you need to define proper lower and upper bounds)
        lower_bound = np.array([100, 100, 100])
        upper_bound = np.array([255, 255, 255])

        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        inverted_mask = cv2.bitwise_not(mask)

        foreground = cv2.bitwise_and(frame_rgb, frame_rgb, mask=inverted_mask)

        result = cv2.addWeighted(foreground, 1, mountain, 0.5, 0)

        cv2.imshow('frame', result)

        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
