#!/usr/bin/python3

import cv2 as cv
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
        cv.resizeWindow('gesture cam', 400,400)
        # Need to move window pos too, top left, center, etc.
        self.run()

    def run(self):
        while True:
            ret_val, frame = cam.read()
            # Draw rectangle when hand is identified/track hand.
            img = cv.rectangle(frame, (384, 0), (510, 128), (0, 255, 0), 3)
            cv.imshow('gesture cam', img)

            # Press ESC to quit.
            if cv.waitKey(1) == 27:
                break

        # Close resources
        cam.release()
        cv.destroyAllWindows()

    # Draws and tracks hand position when identified.
    def drawHandPos(self, frame):
        pass

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sig_handler)
    if len(sys.argv) == 2:
        if sys.argv[1].isdigit() and int(sys.argv[1]) == 1:
            test = camera(1)
        else:
            print("Enter 1 to enable another camera besides default.")
    else:
        test = camera()

