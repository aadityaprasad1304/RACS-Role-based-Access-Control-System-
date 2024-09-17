import face_recognition
import cv2
import os
import numpy as np
import tkinter as tk
import csv
from PIL import Image, ImageTk
import threading
import time
import json

# Set to store recognized names
recognized_names = set()

# Function to update attendance in the CSV file
def update_attendance(name):
    # Check if the name has already been recognized
    if name in recognized_names:
        print(f"{name} already recognized. Skipping...")
        return
    
    # Append the name along with timestamp to the CSV file
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open('Attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp])
    
    # Add the name to the recognized set
    recognized_names.add(name)
    print(f"Attendance recorded for {name} at {timestamp}")

    # Display the image and name for 3 seconds
    display_user_info(name)

def display_user_info(name):
    # Load the user image
    user_image_path = f"{name}.jpg"  # Assuming image path follows the name.jpg convention
    if os.path.exists(user_image_path):
        user_image = Image.open(user_image_path)
        user_image = user_image.resize((160, 160), Image.ANTIALIAS)
        user_photo = ImageTk.PhotoImage(user_image)
        user_label.config(image=user_photo)
        user_label.image = user_photo
        user_label.pack()

        # Display the name
        user_name_label.config(text=name)
        user_name_label.pack()

        # Display attendance marked message
        attendance_label.config(text="Your attendance has been marked")
        attendance_label.pack()

        # Hide the image, name, and attendance message after 3 seconds
        root.after(4000, hide_user_info)
    else:
        print(f"Image not found for {name}")

def hide_user_info():
    user_label.pack_forget()
    user_name_label.pack_forget()
    attendance_label.pack_forget()

# Function to load known faces from JSON file
def load_known_faces():
    try:
        known_faces = {}
        if os.path.exists('known_faces.json'):
            with open('known_faces.json', 'r') as file:
                known_faces_data = json.load(file)
                for name, img_path in known_faces_data.items():
                    image = face_recognition.load_image_file(img_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    known_faces[name] = encoding
        return known_faces
    except Exception as e:
        print(f"Error loading known faces: {e}")
        return {}

# Function to check access based on face recognition
def check_access(rgb_frame):
    try:
        # Load known face encodings and names from JSON file
        known_faces = load_known_faces()

        # If no known faces are loaded, return "Access Denied"
        if not known_faces:
            print("No known faces loaded.")
            return "Access Denied"

        # Extract known encodings and names
        known_encodings = list(known_faces.values())
        known_names = list(known_faces.keys())

        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # If no face is detected, return "Access Denied"
        if len(face_encodings) == 0:
            print("No face detected.")
            return "Access Denied"

        # Loop through each face in the current frame
        for face_encoding in face_encodings:
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, update the name and break the loop
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
                print(f"Match found: {name}")
                update_attendance(name)  # Append name to CSV file
                return f"Welcome {name} - Access Granted"

        # If no match is found among known faces, return "Access Denied"
        print("No match found among known faces.")
        return "Access Denied"

    except Exception as e:
        print(f"Error: {e}")
        return "Access Denied"

# Function to capture video from the camera and update the GUI with real-time feed
def capture_video():
    # Open the video capture device (webcam)
    video_capture = cv2.VideoCapture(0)
    last_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Resize the frame
        frame = cv2.resize(frame, (640, 480))

        # Convert the frame from BGR color to RGB color
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get the access status message
        message = check_access(rgb_frame)

        # Display the message on the GUI
        name_label.config(text=message)

        # Display the frame in the GUI
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk
        panel.config(image=imgtk)

        # Update the GUI window at a certain frame rate (30 FPS)
        current_time = time.time()
        elapsed_time = current_time - last_time
        if elapsed_time >= 1 / 30:
            root.update_idletasks()
            root.update()
            last_time = current_time

        # If 'q' is pressed, break the loop and stop the video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close the OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Function to display the user's image and name for 3 seconds
def display_user_info_thread(name):
    # Load the user image
    user_image_path = f"{name}.jpg"  # Assuming image path follows the name.jpg convention
    if os.path.exists(user_image_path):
        user_image = Image.open(user_image_path)
        user_image = user_image.resize((160, 160), Image.ANTIALIAS)
        user_photo = ImageTk.PhotoImage(user_image)
        user_label.config(image=user_photo)
        user_label.image = user_photo
        user_label.pack()

        # Display the name
        user_name_label.config(text=name)
        user_name_label.pack()

        # Hide the image and name after 3 seconds
        root.after(4000, hide_user_info)
    else:
        print(f"Image not found for {name}")

# Create the tkinter GUI window
root = tk.Tk()
root.title("Face Recognition Attendance System")

# Create a label to display the access status message
name_label = tk.Label(root, text="", font=("Arial", 20))
name_label.pack()

# Create a panel to display the camera feed
panel = tk.Label(root)
panel.pack()

# Create a label to display the recognized user image and name
user_label = tk.Label(root)
user_name_label = tk.Label(root, font=("Arial", 16))

# Create a label to display the attendance message
attendance_label = tk.Label(root, font=("Arial", 16))

# Start capturing video in a separate thread
video_thread = threading.Thread(target=capture_video)
video_thread.daemon = True
video_thread.start()

# Start the tkinter event loop
root.mainloop()
