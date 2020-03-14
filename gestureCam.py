#!/usr/bin/python3

import cv2 as cv
import numpy as np
import signal
import sys

#####################################
# Project 3 : CIS365-01
# Title: Gesture ANN Camera
# Descripiton:
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
        cv.namedWindow("gesture cam", cv.WINDOW_NORMAL)
        cv.resizeWindow('gesture cam', 800, 800)
        # Need to move window pos too, top left, center, etc.
        self.run()
        #frame = cv.imread('hand2.jpg', 0)
        #cv.imshow('Original', frame)
        #self.drawHandPos(frame)

        #i = 0
        #while i < 10:
            #input('Press Enter to capture')
            #return_value, image = cam.read()
            #cv.imwrite('opencv'+str(i)+'.png', image)
            #i += 1
        #cam.release()
        #cv.destroyAllWindows()

    def run(self):
        while True:
            ret_val, frame = cam.read()

            # Draw rectangle when hand is identified/track hand.
            img = cv.rectangle(frame, (100, 100), (300, 350), (0, 255, 0), 3)
            cropFrame = frame[100:350, 100:300]

            hsv = cv.cvtColor(cropFrame, cv.COLOR_BGR2HSV)
            lower_red = np.array([30,150,50]) 
            upper_red = np.array([255,255,180])
            mask = cv.inRange(hsv, lower_red, upper_red) 
            res = cv.bitwise_and(cropFrame, cropFrame, mask= mask) 
            edges = cv.Canny(cropFrame,100,200) 

            cv.imshow('gesture cam', img)
            cv.imshow('cropped', edges)

            # Press ESC to quit.
            if cv.waitKey(1) == 27:
                break

        # Close resources
        cam.release()
        cv.destroyAllWindows()

    # Draws and tracks hand position when identified.
    def drawHandPos(self, frame): 
        ret, thresh = cv.threshold(frame, 10, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        M = cv.moments(cnt)

        hull = [cv.convexHull(c) for c in contours]
        final = cv.drawContours(frame, hull, -1, (255, 255, 255))

        cv.imshow('Thresh', thresh)
        cv.imshow('Convex Hull', frame)
        cv.waitKey(0)

        print(M)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sig_handler)
    if len(sys.argv) == 2:
        if sys.argv[1].isdigit() and int(sys.argv[1]) == 1:
            test = camera(1)
        else:
            print("Enter 1 to enable another camera besides default.")
    else:
        test = camera()

