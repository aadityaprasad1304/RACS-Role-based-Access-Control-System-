# RACS-Role-based-Access-Control-System-


## Overview

This project is a face recognition attendance system that uses OpenCV and the `face_recognition` library to capture video from a webcam, recognize known faces, and mark attendance by saving the names and timestamps to a CSV file. It also uses a Tkinter-based GUI to display the real-time camera feed and additional information about recognized users.

## Project Structure

- **Face Recognition**: The system loads known face encodings from a JSON file and compares them with faces detected in the live video feed. If a face is recognized, the system updates the attendance in a CSV file.
- **GUI**: A Tkinter GUI displays the live video feed, access status (granted or denied), and user information (name and photo) when recognized.
- **Multithreading**: The video capture and GUI run in separate threads to ensure smooth real-time performance.

## Key Components

### Libraries Used

- `face_recognition`: For face detection and face encoding.
- `cv2` (OpenCV): For capturing video and frame manipulation.
- `os`: For handling file paths and image file management.
- `numpy`: For numerical operations and handling frames as arrays.
- `tkinter`: For building the graphical user interface.
- `csv`: For storing attendance records.
- `PIL`: For handling images in the GUI.
- `threading`: For handling video capture in a separate thread.
- `json`: For loading known face encodings.

### Main Components

#### 1. **Updating Attendance**
   The function `update_attendance` is responsible for checking whether a recognized person has already been logged. If not, it adds the person's name and a timestamp to the CSV file.

#### 2. **Displaying User Info**
   The function `display_user_info` shows the recognized user's image and name on the GUI. The image is loaded based on the assumption that it follows a convention like `name.jpg`. The user’s name and attendance confirmation message are also displayed.

#### 3. **Loading Known Faces**
   The `load_known_faces` function loads face encodings from a JSON file (`known_faces.json`). This file contains the mapping between user names and their respective image file paths. Each face is encoded and stored for future recognition.

#### 4. **Face Recognition and Access Control**
   The `check_access` function compares faces in the current frame with the known faces. If a match is found, the user’s name is returned, and attendance is updated. If no match is found, "Access Denied" is returned.

#### 5. **Capturing Video**
   The `capture_video` function continuously captures video from the webcam, processes each frame by resizing and converting it, and updates the Tkinter GUI with the current frame. It calls `check_access` to determine whether a recognized face is present in the frame.

#### 6. **Multithreading**
   The video capture runs on a separate thread using the `threading` library to ensure that the Tkinter GUI remains responsive.

## Files Used

1. **Attendance.csv**: The CSV file where attendance records (name and timestamp) are stored.
2. **known_faces.json**: A JSON file that contains mappings of names to their image paths for face encoding.

## GUI Structure

- **Real-time Camera Feed**: The video feed is displayed using a `Label` widget that updates continuously.
- **Name and Image**: When a face is recognized, the user’s image and name are displayed for 3 seconds.
- **Attendance Message**: A message indicating that attendance has been marked is shown when a face is recognized.
