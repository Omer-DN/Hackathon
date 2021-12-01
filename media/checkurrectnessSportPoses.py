import cv2
import mediapipe as mp
import math

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils


# The function received the image and the locate pose in the image as parameters
#  and return the image after proccessing and the landmarks in the image
def detect_pose(image, pose):
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

    # Return the output image and the found landmarks.
    return output_image, landmarks


def calculate_angle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle


def right_hand_angles(landmarks, mp_pose):
    # Get the angle between the right shoulder, elbow and wrist points.
    right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # Get the angle between the right hip, shoulder and elbow points.
    right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    return right_shoulder_angle, right_elbow_angle


def left_hand_angles(landmarks, mp_pose):
    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # Get the angle between the left elbow, shoulder and hip points.
    left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    return left_shoulder_angle, left_elbow_angle


def serratus_strech(limb1, limb2):
    if abs(limb1 - 117.09) < 10 and abs(limb2 - 206.36) < 10:
        return 1

    elif abs(limb1 - 188) < 10 and abs(limb2 - 213) < 10:
        return 2

    else:
        return 0


def main():
    counter = 0
    arrivePose1 = False

    # Setup Pose function for video.
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)

    # Initialize a resizable window.
    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():

        # Read a frame.
        ok, frame = camera_video.read()

        # Check if frame is not read properly.
        if not ok:
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Get the width and height of the frame
        frame_height, frame_width, _ = frame.shape

        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        # Perform Pose landmark detection.
        frame, landmarks = detect_pose(frame, pose_video)

        # Check if the landmarks are detected.
        if landmarks:
            right_shoulder_angle, right_elbow_angle = right_hand_angles(landmarks, mp_pose)
            left_shoulder_angle, left_elbow_angle = left_hand_angles(landmarks, mp_pose)
            #####################################################################################
            print("right shoulder angle: " + str(right_shoulder_angle))
            print("right_elbow_angle: " + str(right_elbow_angle))
            #####################################################################################
            pose_score = serratus_strech(right_shoulder_angle, right_elbow_angle)
            pose_score2 = serratus_strech(left_shoulder_angle, left_elbow_angle)

            label = str(counter) + " times"
            color = (75, 255, 255)
            cv2.putText(frame, label, (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            # Perform the Pose Classification.
            if pose_score == 0 or pose_score2 == 0:
                label = 'Wrong Pose'
                color = (0, 0, 255)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.imshow('Pose Classification', frame)

            elif pose_score == 1 or pose_score2 == 1:
                arrivePose1 = True
                # Display the frame.
                label = 'Pose number 1'
                color = (0, 255, 0)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.imshow('Pose Classification', frame)

            elif pose_score == 2 or pose_score2 == 2:
                if arrivePose1:
                    counter += 1
                    arrivePose1 = False
                # Display the frame.
                label = 'Pose number 2'
                color = (0, 255, 0)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.imshow('Pose Classification', frame)

        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed.
        if (k == 27):
            # Break the loop.
            break

    # Release the VideoCapture object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
