#!/usr/bin/python3

import cv2 as cv
import signal
import sys
import tensorflow.keras
import numpy as np
import pyautogui
#import profile

"""
Project 3 | CIS365-01
Title: CNN Gesture Camera
Date: 2/29/2020
Authors: Matt Shampine, Nabeel Vali
"""

np.set_printoptions(suppress=True)

# Load our trained model to be able to make predictions on image input
model = tensorflow.keras.models.load_model('my_model_final.h5')


def sig_handler():
    """
    Handles CTRL-C interrupt to cleanly terminate the program
    :return: None
    """
    print("\nNow closing resources.")
    cam.release()
    cv.destroyAllWindows()
    exit(0)


class camera():
    """
    Initializes the users camera via OpenCV, passes frames from the camera
    into the trained model to generate a gesture prediction. The prediction
    is used to control what computer action is taken.
    """

    def __init__(self, camNum=0):
        """
        Constructor method that establishes camera file descriptor,
        coordinates for drawing on frame, and flag variables used in helper methods.
        :param camNum: Integer that represents which camera gets connected
            when there is more than one present.
        """

        # Global - so the sighandler can close resources upon ^C.
        global cam
        cam = cv.VideoCapture(camNum)
        # Gets the default camera resolution.
        self.width = cam.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
        # print("w:", self.width, "h:", self.height)

        # Calculate center screen (x, y) and text position (x, y).
        self.center_x = (int(self.width / 2))
        self.center_y = (int(self.height / 2))
        self.center = (self.center_x, self.center_y)
        self.txt_center = (self.center_x - 75, self.center_y + 230)

        # Rectangles are drawn with two different diagonal (x,y) coordinates.
        self.box_x1 = self.center_x - 125
        self.box_y1 = self.center_y - 180
        self.pos1 = (self.box_x1, self.box_y1)
        self.box_x2 = self.center_x + 125
        self.box_y2 = self.center_y + 180
        self.pos2 = (self.box_x2, self.box_y2)

        # Class level flags to keep track of detected gestures
        self.handFlag = 0
        self.fistFlag = 0

        # Note: (G, B, R) not (R, G, B).
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)

        self.run()

    def run(self):
        """
        This method executes on program start, it loops infinitely until user
        interruption. A connection to the webcam is created and frames are
        passed to the model for a prediction. Helper methods are then 
        called to generate computer actions and illustrations within the
        application window.
        :return: None
        """
        while True:
            ret_val, frame = cam.read()

            if ret_val == False:
                print("Error reading frame, webcam may be disconnected.")
                exit(-1)

            # Save an unedited frame for CNN consumption.
            saveFrame = frame.copy()

            # Resize the input frame to 224x224, this size needed by the model
            frame1 = cv.resize(saveFrame, (224, 224))
            cv.imshow('Input Frame', frame1)

            # Reshape the frame into a 3 dimensional array that can be understood by the CNN
            frame1 = frame1.reshape(1, 224, 224, 3)
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image_array = np.asarray(frame1)

            # Normalize/Down-sample our image, then pass it into the model as an array
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            prediction = model.predict(data)

            #Predictions are returned in array, grab the index of the max prediction
            maxVal = np.argmax(prediction)

            print(prediction)

            # Chooses rect color based on if a hand is recognized (maxVal > 0).
            if maxVal > 0:
                self.overlayRect(frame, True, maxVal)
            else:
                self.overlayRect(frame, False, maxVal)

            cv.imshow('feed', frame)

            # Press ESC to quit.
            if cv.waitKey(1) == 27:
                break

        # Close resources.
        cam.release()
        cv.destroyAllWindows()

    def overlayRect(self, frame, isHand, gest):
        """
        Draws rectangle on video feed to aid in hand placement. Performs two
        different keyboard actions based off of hand gestures.
        :param frame: Current video frame.
        :param isHand: Boolean that is true when a hand is detected in frame.
        :param gest: An integer that maps to gestures:
            0 - wall, 1 - hand, 2 - fist
        :return: None
        """
        set_color = self.green
        text = ""
        if isHand and gest == 1:
            text = "Hand"
            self.moveHand()
        elif isHand and gest == 2:
            text = " Fist"
            self.moveFist()
        else:
            set_color = self.red
            # Reset the gesture detected flags to 0
            self.handFlag = 0
            self.fistFlag = 0

        # Draw center circle.
        cv.circle(frame, self.center, 8, set_color, -1)
        # Draw ROI (region of interest) on frame.
        cv.rectangle(frame, self.pos1, self.pos2, set_color, 3)
        cv.putText(frame, text, self.txt_center, cv.FONT_HERSHEY_SIMPLEX, 2, self.green, 2)

    def moveHand(self):
        """
        When a hand is detected, this method calls PyAutoGUI to imitate
        a keyboard press. 'space' is pressed and handFlag is set to true.
        :return: None
        """
        if self.handFlag != 1:
            pyautogui.press('space')
            self.handFlag = 1

    def moveFist(self):
        """
        When a fist is detected, this method calls PyAutoGUI to imitate
        a keyboard press. 'Ctrl-Right' is skip a song in Spotify.
        FistFlag is set to true.
        :return:
        """
        if self.fistFlag != 1:
            pyautogui.hotkey('ctrl', 'right')
            self.fistFlag = 1


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sig_handler)
    if len(sys.argv) == 2:
        if sys.argv[1].isdigit() and int(sys.argv[1]) == 1:
            #profile.run('camera(1)')
            camera(1)
        else:
            print("Enter 1 to enable another camera besides default.")
    else:
        camera()

