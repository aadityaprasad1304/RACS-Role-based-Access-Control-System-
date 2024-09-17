import os
import json
import face_recognition

def encode_faces_in_directory(directory):
    encoded_faces = {}
    for person in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, person)):
            encoded_faces[person] = []
            for image_file in os.listdir(os.path.join(directory, person)):
                image_path = os.path.join(directory, person, image_file)
                face_image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(face_image)
                if len(face_encodings) > 0:
                    encoded_faces[person].append(face_encodings[0].tolist())
    return encoded_faces

def main():
    known_faces_directory = "known_faces"
    encoded_faces = encode_faces_in_directory(known_faces_directory)
    
    with open("encoded_faces.json", "w") as json_file:
        json.dump(encoded_faces, json_file)

if __name__ == "__main__":
    main()
