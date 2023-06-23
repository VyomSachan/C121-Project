# import cv2 to capture videofeed
import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the background image
background = cv2.imread('colorful-tree-valley.jpg')

# resizing the background image as 640 X 480
background = cv2.resize(background , (640 , 480))

while True:
    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([95, 40, 20])
        upper_bound = np.array([220, 210, 170])

        # thresholding image
        person_face = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # inverting the mask
        person_face = cv2.bitwise_not(person_face)

        # bitwise and operation to extract foreground / person
        final_person = cv2.bitwise_and(frame, frame , mask = person_face)

        # final image
        frame = np.where(final_person  ==  0 , background , final_person)

        # show it
        cv2.imshow('frame' , frame)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()