import tkinter as tk
from tkinter.font import Font
from tkinter.ttk import Label
import cv2
from PIL import Image, ImageTk
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
# from pydub import AudioSegment
# from playsound import playsound
from gtts import gTTS
import os
import time
import threading
import pygame 

class WebcamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SIGN LANGUAGE TRANSCRIBER")
        
        self.root.geometry("700x720")
        title_font = Font(family="Helvetica", size=20, weight="bold")  # Define font properties
        self.title_label = Label(self.root, text="SIGN LANGUAGE TRANSCRIBER", font=title_font)
        self.title_label.pack()
       
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        # create a slightly bigger textbox with a title
        self.textbox = tk.Text(self.root, width=50, height=3, font=("Helvetica", 14))
        self.textbox.pack()
        self.textbox.insert(tk.END, "WELCOME")

        # Create a button to start recognition
        self.start_button = tk.Button(self.root, text="Start Recognition", command=self.start_recognition, font=("Helvetica", 14), bg="blue", fg="white", borderwidth=3, relief="raised", padx=10, pady=5)
        self.start_button.pack(pady=(20, 5))  # Add padding between buttons

        self.play_button = tk.Button(self.root, text="Play Sound", command=self.play_sound, font=("Helvetica", 14), bg="green", fg="white", borderwidth=3, relief="raised", padx=10, pady=5)
        self.play_button.pack(pady=5)

        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("model/keras_model.h5", "model/labels.txt")
        self.offset = 20
        self.imgSize = 400
        self.folder = "Data/outputs"
        self.labels = ['hello', 'thankyou', 'yes', 'no', 'iloveyou', 'please']
        self.current_gesture = None  # Variable to store the current recognized gesture
        self.processing = True  # Flag to control whether to continue processing gestures

        self.update_image()

    def update_image(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        # convert the BGR frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # convert the numpy array to a PIL image
        image = Image.fromarray(frame)

        # convert the PIL image to a Tkinter PhotoImage
        self.photo = ImageTk.PhotoImage(image)

        # display the image on the canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # repeat the update in 10 milliseconds
        self.root.after(10, self.update_image)

    def start_recognition(self):
        # Start recognition in a separate thread
        threading.Thread(target=self.recognition_thread).start()

    def recognition_thread(self):
        while self.processing:
            ret, frame = self.capture.read()
            if not ret:
                break

            gesture = self.process_image_and_sound(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def process_image_and_sound(self, img):
        imgOutput = img.copy()
        hands, _ = self.detector.findHands(img)
        gesture = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = self.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                wGap = math.ceil((self.imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = self.classifier.getPrediction(imgWhite)
            else:
                k = self.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                hGap = math.ceil((self.imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = self.classifier.getPrediction(imgWhite)

            gesture = self.labels[index]

            cv2.putText(imgOutput, gesture, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset), (0, 255, 0), 2)

            self.update_textbox(gesture)
            self.text_to_speech(gesture)
            self.play_sound()  
            os.remove("output.mp3")

        cv2.imshow("Image", imgOutput)
        return gesture


    def text_to_speech(self, text):
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        


    def play_sound(self):
            sound_file = "output.mp3"
            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            threading.Thread(target=self.continue_processing).start()

    def continue_processing(self):
        time.sleep(0.5)  # Adjust the sleep time if necessary
        self.processing = True

    def update_textbox(self, gesture):
        self.textbox.delete(1.0, tk.END)
        self.textbox.insert(tk.END, gesture)
        self.text_to_speech(gesture)  # Speak the recognized gesture
    


root = tk.Tk()
app = WebcamGUI(root)
root.mainloop()
