
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from pyparsing import results


# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils


def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Otherwise
    else:

        # Return the output image and the found landmarks.
        return output_image, landmarks


def main():
    print("Start")

    # Read an image from the specified path.
    sample_img = cv2.imread('../media/sample4.jpg')

    # Specify a size of the figure.
    plt.figure(figsize=[10, 10])

    # Display the sample image, also convert BGR to RGB for display.
    plt.title("Sample Image")
    plt.axis('off')
    plt.imshow(sample_img[:, :, ::-1])
    plt.show()

    # Perform pose detection after converting the image into RGB format.
    results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

    # Check if any landmarks are found.
    if results.pose_landmarks:

        # Iterate two times as we only want to display first two landmarks.
        for i in range(2):
            # Display the found normalized landmarks.
            print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')

    image_height, image_width, _ = sample_img.shape

    # Check if any landmarks are found.
    if results.pose_landmarks:

        # Iterate two times as we only want to display first two landmark.
        for i in range(2):
            # Display the found landmarks after converting them into their original scale.
            print(f'{mp_pose.PoseLandmark(i).name}:')
            print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
            print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')
            print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')
            print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')

    # Create a copy of the sample image to draw landmarks on.
    img_copy = sample_img.copy()

    # Check if any landmarks are found.
    if results.pose_landmarks:
        # Draw Pose landmarks on the sample image.
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Specify a size of the figure.
        fig = plt.figure(figsize=[10, 10])

        # Display the output image with the landmarks drawn, also convert BGR to RGB for display.
        plt.title("Output");
        plt.axis('off');
        plt.imshow(img_copy[:, :, ::-1]);
        plt.show()

    # Plot Pose landmarks in 3D.
    mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Read another sample image and perform pose detection on it.
    image = cv2.imread('../media/sample1.jpg')
    detectPose(image, pose, display=True)


if __name__ == "__main__":
    main()
