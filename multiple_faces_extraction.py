import cv2
import face_recognition
import os

# Set up input image path and output directory
input_image_path = 'dataset//fake//01_11__meeting_serious__9OM3VE0Y//frame_0020.jpg'  # Replace with your image file path
output_dir = 'outputImages/extracted_faces/'         # Directory to save cropped face images
os.makedirs(output_dir, exist_ok=True)

# Load the image
image = cv2.imread(input_image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect face locations in the image
face_locations = face_recognition.face_locations(rgb_image)

# Loop through each face found
for i, (top, right, bottom, left) in enumerate(face_locations):
    # Extract face region
    face_image = image[top:bottom, left:right]

    # Save the cropped face
    face_filename = os.path.join(output_dir, f"face_{i+1}.jpg")
    cv2.imwrite(face_filename, face_image)
    print(f"Face {i+1} saved as {face_filename}")

print("Face extraction complete.")
