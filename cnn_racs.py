CODE:

# Import necessary libraries
import cv2  # OpenCV library for video capturing and image processing
import os  # Provides functions for interacting with the operating system
import numpy as np  # Provides support for arrays and matrices
import tkinter as tk  # Used to create a graphical user interface (GUI)
import csv  # Used to read from and write to CSV files
from PIL import Image, ImageTk  # Pillow library for image manipulation
import threading  # Used to run multiple operations concurrently
import time  # Provides time-related functions
import json  # Used for parsing JSON data
from facenet_pytorch import MTCNN, InceptionResnetV1  # Face detection and embedding model
import torch  # PyTorch library for deep learning

# Initialize the MTCNN face detector and the InceptionResnetV1 face embedding model
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise use CPU
mtcnn = MTCNN(keep_all=True, device=device)  # Initialize MTCNN for face detection
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Initialize InceptionResnetV1 for face embeddings

# Set to store recognized names
recognized_names = set()  # Keeps track of names that have already been recorded to avoid duplicates

def update_attendance(name):
    """
    Update the attendance CSV file and display user information on the GUI.

    Args:
        name (str): The name of the recognized person.
    """
    # Check if the name has already been recognized
    if name in recognized_names:
        print(f"{name} already recognized. Skipping...")
        return
    
    # Append the name along with timestamp to the CSV file
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Get current timestamp
    with open('Attendance.csv', 'a', newline='') as file:  # Open the CSV file in append mode
        writer = csv.writer(file)  # Create a CSV writer object
        writer.writerow([name, timestamp])  # Write the name and timestamp to the file
    
    # Add the name to the recognized set
    recognized_names.add(name)  # Update the set with the recognized name
    print(f"Attendance recorded for {name} at {timestamp}")

    # Display the image and name for 3 seconds
    display_user_info(name)  # Show the user's information on the GUI

def display_user_info(name):
    """
    Display the user's image and name on the GUI for 3 seconds.

    Args:
        name (str): The name of the recognized person.
    """
    # Load the user image
    user_image_path = f"{name}.jpg"  # Assuming image path follows the name.jpg convention
    if os.path.exists(user_image_path):  # Check if the image file exists
        user_image = Image.open(user_image_path)  # Open the image file
        user_image = user_image.resize((160, 160), Image.ANTIALIAS)  # Resize the image
        user_photo = ImageTk.PhotoImage(user_image)  # Convert image to PhotoImage for Tkinter
        user_label.config(image=user_photo)  # Update the image in the label
        user_label.image = user_photo  # Keep a reference to avoid garbage collection
        user_label.pack()  # Display the image label

        # Display the name
        user_name_label.config(text=name)  # Update the label with the user's name
        user_name_label.pack()  # Display the name label

        # Display attendance marked message
        attendance_label.config(text="Your attendance has been marked")  # Update the attendance message
        attendance_label.pack()  # Display the attendance label

        # Hide the image, name, and attendance message after 3 seconds
        root.after(4000, hide_user_info)  # Schedule hide_user_info to be called after 4 seconds
    else:
        print(f"Image not found for {name}")  # Print an error message if the image is not found

def hide_user_info():
    """
    Hide the user's image, name, and attendance message from the GUI.
    """
    user_label.pack_forget()  # Remove the image label from the GUI
    user_name_label.pack_forget()  # Remove the name label from the GUI
    attendance_label.pack_forget()  # Remove the attendance message from the GUI

def load_known_faces():
    """
    Load known face embeddings from a JSON file.

    Returns:
        dict: A dictionary with names as keys and face embeddings as values.
    """
    try:
        known_faces = {}  # Initialize an empty dictionary to store known faces
        if os.path.exists('known_faces.json'):  # Check if the JSON file exists
            with open('known_faces.json', 'r') as file:  # Open the JSON file for reading
                known_faces_data = json.load(file)  # Load the JSON data
                for name, img_path in known_faces_data.items():  # Iterate through the JSON data
                    image = Image.open(img_path).convert('RGB')  # Open and convert image to RGB
                    face = mtcnn(image)  # Detect faces in the image
                    if face is not None:  # Check if any faces were detected
                        embedding = model(face.to(device)).detach().cpu().numpy()[0]  # Compute face embedding
                        known_faces[name] = embedding  # Store the embedding in the dictionary
        return known_faces  # Return the dictionary of known faces
    except Exception as e:
        print(f"Error loading known faces: {e}")  # Print error message if something goes wrong
        return {}

def check_access(rgb_frame):
    """
    Check if any face in the current frame matches a known face.

    Args:
        rgb_frame (numpy.ndarray): The current video frame in RGB format.

    Returns:
        str: Access status message ("Access Granted" or "Access Denied").
    """
    try:
        # Convert the frame to PIL image
        image = Image.fromarray(rgb_frame)  # Convert the numpy array to a PIL image
        
        # Detect faces and extract embeddings
        faces = mtcnn(image)  # Detect faces in the image
        if faces is None:  # Check if no faces were detected
            print("No face detected.")
            return "Access Denied"
        
        embeddings = model(faces.to(device)).detach().cpu().numpy()  # Compute embeddings for detected faces
        
        known_faces = load_known_faces()  # Load known face embeddings
        if not known_faces:  # Check if no known faces were loaded
            print("No known faces loaded.")
            return "Access Denied"
        
        known_encodings = list(known_faces.values())  # Get the embeddings of known faces
        known_names = list(known_faces.keys())  # Get the names of known faces
        
        # Compare each face embedding with known faces
        for embedding in embeddings:  # Loop through each face embedding
            distances = [np.linalg.norm(embedding - known_encoding) for known_encoding in known_encodings]  # Compute distances
            min_distance = min(distances)  # Get the minimum distance
            if min_distance < 1.0:  # Check if the distance is below the threshold
                name = known_names[distances.index(min_distance)]  # Get the name of the matching face
                print(f"Match found: {name}")
                update_attendance(name)  # Record attendance
                return f"Welcome {name} - Access Granted"
        
        print("No match found among known faces.")
        return "Access Denied"
    
    except Exception as e:
        print(f"Error: {e}")  # Print error message if something goes wrong
        return "Access Denied"

def capture_video():
    """
    Capture video from the webcam, process frames, and update the GUI.
    """
    # Open the video capture device (webcam)
    video_capture = cv2.VideoCapture(0)  # Initialize video capture
    last_time = time.time()  # Record the current time

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()  # Read a frame from the webcam

        # Resize the frame
        frame = cv2.resize(frame, (640, 480))  # Resize the frame to a fixed size

        # Convert the frame from BGR color to RGB color
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Get the access status message
        message = check_access(rgb_frame)  # Check if access is granted or denied

        # Display the message on the GUI
        name_label.config(text=message)  # Update the message label

        # Display the frame in the GUI
        img = Image.fromarray(rgb_frame)  # Convert the RGB frame to a PIL image
        imgtk = ImageTk.PhotoImage(image=img)  # Convert the PIL image to PhotoImage for Tkinter
        panel.imgtk = imgtk  # Keep a reference to avoid garbage collection
        panel.config(image=imgtk)  # Update the image in the panel

        # Update the GUI window at a certain frame rate (30 FPS)
        current_time = time.time()  # Get the current time
        elapsed_time = current_time - last_time  # Calculate elapsed time
        if elapsed_time >= 1 / 30:  # Check if it's time to update the GUI
            root.update_idletasks()  # Update the GUI tasks
            root.update()  # Update the GUI window
            last_time = current_time  # Update the last time

        # If 'q' is pressed, break the loop and stop the video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed
            break

    # Release the video capture device and close the OpenCV windows
    video_capture.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Create the tkinter GUI window
root = tk.Tk()  # Initialize the main GUI window
root.title("Face Recognition Attendance System")  # Set the window title

# Create a label to display the access status message
name_label = tk.Label(root, text="", font=("Arial", 20))  # Initialize the label for access messages
name_label.pack()  # Display the label

# Create a panel to display the camera feed
panel = tk.Label(root)  # Initialize the label for camera feed
panel.pack()  # Display the label

# Create a label to display the recognized user image and name
user_label = tk.Label(root)  # Initialize the label for user image
user_name_label = tk.Label(root, font=("Arial", 16))  # Initialize the label for user name

# Create a label to display the attendance message
attendance_label = tk.Label(root, font=("Arial", 16))  # Initialize the label for attendance message

# Start capturing video in a separate thread
video_thread = threading.Thread(target=capture_video)  # Create a new thread for video capture
video_thread.daemon = True  # Set the thread as a daemon so it exits when the main program exits
video_thread.start()  # Start the video capture thread

# Start the tkinter event loop
root.mainloop()  # Run the Tkinter event loop to keep the GUI responsive



Summary of Changes:
1.	Imports:
○	Added imports for facenet_pytorch, torch, and necessary modules.
2.	Model Initialization:
○	Initialized the MTCNN face detector and InceptionResnetV1 model for face recognition.
3.	Face Detection and Recognition:
○	Updated load_known_faces and check_access functions to use FaceNet for detecting and recognizing faces.
4.	Video Capture and GUI:
○	Kept the capture_video and GUI setup largely unchanged, ensuring compatibility with the updated face recognition logic.
PyTorch, MTCNN & InceptionResnetV1

1. MTCNN (Multi-task Cascaded Convolutional Networks)
Purpose:
●	MTCNN is used for face detection and alignment. It can locate multiple faces in an image and provide precise bounding boxes around them.
Functionality:
●	Detection: Identifies where faces are in an image.
●	Alignment: Provides landmarks (e.g., eyes, nose, mouth) to help align the face for further processing.
How It Works:
●	MTCNN consists of three stages:
1.	Proposal Network (P-Net): Generates candidate face regions.
2.	Refinement Network (R-Net): Refines the candidate regions and performs face classification.
3.	Output Network (O-Net): Further refines the regions and estimates facial landmarks.
Usage in Code:
●	mtcnn = MTCNN(keep_all=True, device=device): Initializes the MTCNN object to detect and align all faces in an image. The device parameter specifies whether to use a GPU or CPU.
2. InceptionResnetV1
Purpose:
●	InceptionResnetV1 is a deep convolutional neural network used for face recognition. It extracts high-dimensional face embeddings that represent the features of a face in a compact form.
Functionality:
●	Face Embeddings: Converts a face image into a numerical vector (embedding) that captures the unique features of the face. These embeddings can be compared to determine if two faces are the same or different.
How It Works:
●	InceptionResnetV1 combines the strengths of Inception networks and Residual networks. It utilizes multiple convolutional layers and residual connections to learn complex features of the face.
Usage in Code:
●	model = InceptionResnetV1(pretrained='vggface2').eval().to(device): Initializes the InceptionResnetV1 model with pre-trained weights from the VGGFace2 dataset. The model is set to evaluation mode and moved to the specified device (GPU or CPU).
Summary
●	MTCNN is used for detecting faces and their landmarks in images, which helps in accurately cropping and aligning faces.
●	InceptionResnetV1 is used to generate embeddings for the detected faces, which are then used to recognize or compare faces based on their unique features.

FaceNet and CNN:

FaceNet Overview
FaceNet is designed to map face images to a compact Euclidean space where distances directly correspond to face similarity. The model produces embeddings (feature vectors) of fixed length, and the distances between these embeddings can be used to measure the similarity between faces. FaceNet uses a CNN architecture, often based on Inception or ResNet models, to extract features from face images.
CNN Layers in FaceNet
FaceNet's architecture can vary, but it typically involves the following types of layers:
1.	Convolutional Layers
○	Function: Apply convolutional filters to the input image to extract local features. Each filter detects specific patterns like edges, textures, or corners.
○	Example: A convolutional layer might use a filter of size 3x3 to slide over the image and create feature maps.
2.	Activation Layers (ReLU)
○	Function: Introduce non-linearity into the model. The Rectified Linear Unit (ReLU) activation function is commonly used, which sets all negative values to zero and keeps positive values unchanged.
○	Example: After applying convolutional filters, ReLU activation ensures that the network can learn complex patterns and relationships.
3.	Pooling Layers
○	Function: Reduce the spatial dimensions of the feature maps while retaining important features. This helps in reducing computational complexity and controlling overfitting.
○	Types:
■	Max Pooling: Takes the maximum value from a feature map region.
■	Average Pooling: Takes the average value from a feature map region.
○	Example: A 2x2 max pooling layer reduces the dimensions of the feature map by selecting the maximum value in each 2x2 region.
4.	Normalization Layers (Batch Normalization)
○	Function: Normalize the outputs of the previous layer to stabilize and speed up training. This layer adjusts the mean and variance of the features to ensure stable training.
○	Example: Batch normalization is applied after convolutional layers and before activation functions.
5.	Fully Connected Layers (Dense Layers)
○	Function: Connect all neurons from the previous layer to every neuron in the current layer. They aggregate features learned by convolutional layers to make final predictions or generate embeddings.
○	Example: In FaceNet, the final fully connected layer generates a fixed-size embedding vector from the learned features.
6.	Embedding Layer
○	Function: Outputs the face embedding vector of fixed length. This vector is a numerical representation of the face and is used for face comparison.
○	Example: The embedding vector might be 128 or 512 dimensions, depending on the architecture.
Example of a FaceNet CNN Architecture
FaceNet often uses an architecture based on Inception or ResNet networks. Here's a simplified example of what the layers might look like in a FaceNet-like CNN:
1.	Initial Convolutional Layers:
○	Multiple convolutional layers with ReLU activation to extract low-level features.
2.	Inception or ResNet Blocks:
○	Inception Block: Includes multiple convolutional filters of different sizes, concatenated together to capture different feature scales.
○	ResNet Block: Utilizes residual connections to allow the network to learn more complex features by adding input directly to the output.
3.	Fully Connected Layers:
○	Aggregates features from convolutional layers to create a compact representation.
4.	Embedding Layer:
○	Produces the final face embedding vector used for face recognition.
Summary
●	Convolutional Layers: Extract features from the input image.
●	Activation Layers (ReLU): Introduce non-linearity.
●	Pooling Layers: Reduce dimensionality and retain key features.
●	Normalization Layers: Stabilize and speed up training.
●	Fully Connected Layers: Aggregate features and create embeddings.
●	Embedding Layer: Produces a fixed-length vector representing the face.
This combination of layers allows FaceNet to learn detailed and discriminative features of faces, making it effective for face recognition and verification tasks.

