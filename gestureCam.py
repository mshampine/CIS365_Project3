#!/usr/bin/python3

import cv2 as cv
import tensorflow.keras
import numpy as np
import signal
import sys

#####################################
# Project 3 : CIS365-01
# Title: Gesture ANN Camera
# Description:
# Date: 2/29/2020
# Authors: Matt Shampine, Nabeel Vali
#####################################

def sig_handler(signal, frame):
    print("\nNow closing resources.")
    cam.release()
    cv.destroyAllWindows()
    exit(0)

class camera():
    def __init__(self, camNum=0):
        global cam # Global -- so the sighandler can close resources upon ^C.
        cam = cv.VideoCapture(camNum)
        cv.namedWindow("camera feed") #cv.WINDOW_NORMAL)
        cv.resizeWindow('gesture cam', 800, 800)
        cv.moveWindow('camera feed', 200, 200)
        self.run()

    # Something similar to Handy's skin capture histogram method.
    # Calibrate skin.
    def skinColorHist():
        pass

    def run(self):
        # Defines min and max YCrCb skin color. 
        # Right now this is hard-coded, but my idea was to 
        # calibrate these values at the start of the program.
        # That way these values match each person's skin color.
        minVal = np.array([0,133,77], np.uint8)
        maxVal = np.array([235,173,127], np.uint8)

        while True:
            retVal, frame = cam.read()

            # Draw ROI (region of interest) for skin detection.
            img = cv.rectangle(frame, (100, 100), (300, 350), (0, 255, 0), 3)

            # 5x5 pixel padding, so boarder isnt in cropped image.
            cropFrame = frame[105:345, 105:295]

            # Image preproccessing.
            preImage = cv.GaussianBlur(cropFrame, (5,5), 0)

            # Convert frame to YCrCb, contains more usable information.
            imageYCrCb = cv.cvtColor(preImage, cv.COLOR_BGR2YCR_CB)

            # Find pixels in range of targeted skin value.
            skinRegion = cv.inRange(imageYCrCb, minVal, maxVal)

            # Segment hand pixels from rest of frame.
            skinYCrCb = cv.bitwise_and(preImage, preImage, mask = skinRegion)
            
            # Now convert frame to gray to determine bright spots.
            grayFrame = cv.cvtColor(preImage, cv.COLOR_BGR2GRAY)
            brtThresh = cv.threshold(grayFrame, 200, 255, cv.THRESH_BINARY)[1]

            # More image processing.
            brtThresh = cv.erode(brtThresh, None, iterations=2)
            brtThresh = cv.dilate(brtThresh, None, iterations=4)

            # Not exactly sure if this line improves image processing.
            brtThresh = cv.morphologyEx(brtThresh, cv.MORPH_CLOSE, None)

            # Combine skin with highlighted skin.
            skinHighlights = cv.bitwise_and(cropFrame, cropFrame, mask = brtThresh)
            skin = skinYCrCb | skinHighlights
            skinCopy = skin.copy()

            grayImg = cv.cvtColor(skin, cv.COLOR_BGR2GRAY)
            ret, thresh2 = cv.threshold(grayImg, 127, 255, 0)
            contours, hierarchy = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # Draws contour outline of matching skin pixels.
            skin = cv.drawContours(skin, contours, -1, (0,255,0), 3)

            # Get frame with background and skin with contours applied.
            skinROI = cropFrame | skin

            # Replace the current frame's ROI with contour applied.
            frame[105:345, 105:295,:] = skinROI

            cv.imshow('camera feed', frame)
            cv.imshow('highlights', skinHighlights)
            cv.imshow('skin', skinCopy)

            # Press ESC to quit.
            if cv.waitKey(1) == 27:
                break

        # Close resources
        cam.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sig_handler)
    if len(sys.argv) == 2:
        if sys.argv[1].isdigit() and int(sys.argv[1]) == 1:
            test = camera(1)
        else:
            print("Enter 1 to enable another camera besides default.")
    else:
        test = camera()

