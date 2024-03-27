import cv2
import mediapipe as mp
import math

# variables used for diff purpose in pose detection
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Define the threshold angle for correct posture
CORRECT_POSTURE_THRESHOLD = {
    'shoulder': 160,
    'hip': 30,
    'knee': 40,
    'neck': 150,
    'upper_neck': 40
}


# Define a function to calculate the angle between two lines
def calculate_angle(point1, point2, point3):
    # Extract the x, y, and z coordinates of each landmark
    x1, y1, z1 = point1.x, point1.y, point1.z
    x2, y2, z2 = point2.x, point2.y, point2.z
    x3, y3, z3 = point3.x, point3.y, point3.z

    # Calculate the vectors for the lines
    vector1 = [x2 - x1, y2 - y1]
    vector2 = [x3 - x2, y3 - y2]

    # Calculate the dot product of the vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate the magnitudes of the vectors
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate the angle in radians
    angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)
    angle_deg = round(angle_deg)
    return angle_deg


# Define a function to check the posture
def check_posture(pose_landmarks):
    # Calculate the angle between the shoulders and elbow
    left_neck_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_EAR],
                                      pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER],
                                      pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW])

    neck_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER],
                                 pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_EAR],
                                 pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER])

    # Calculate the angle between the shoulders and hips

    left_shoulder_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW],
                                          pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER],
                                          pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP])

    # Calculate the angle between the hips and knees
    left_hip_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER],
                                     pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP],
                                     pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE])

    # Calculate the angle between the left and right knees and ankles

    left_knee_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP],
                                      pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE],
                                      pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ANKLE])

    # Check if the posture is correct
    is_posture_correct = all([left_shoulder_angle >= CORRECT_POSTURE_THRESHOLD['shoulder'],
                              neck_angle >= CORRECT_POSTURE_THRESHOLD['neck'],
                              left_neck_angle <= CORRECT_POSTURE_THRESHOLD['upper_neck'],
                              left_hip_angle <= CORRECT_POSTURE_THRESHOLD['hip'],
                              left_knee_angle <= CORRECT_POSTURE_THRESHOLD['knee']])
    print(neck_angle, "Neck Angle ")
    print(left_neck_angle, "Left Neck Angle ")
    print(left_shoulder_angle, "Left Shoulder Angle ")
    print(left_hip_angle, "Left Hip Angle ")
    print(left_knee_angle, "left Knee Angle ")
    if is_posture_correct:
        print("Posture is correct")
    else:
        print("Posture is incorrect")


# it reads the image file and convert it to RGB from BGR and process the landmarks and draw the landmarks on the image
img = cv2.imread("side_images/image18.png", 1)

if img is None:
    print("Error loading the image.")
    exit(1)

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = pose.process(imgRGB)

# Call the check_posture function with the pose landmarks
check_posture(result.pose_landmarks)

# print(result.pose_landmarks)

if result.pose_landmarks:
    mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # for id, lm in enumerate(result.pose_landmarks.landmark):
    #     h, w, c = img.shape
    #     # print(id, lm)
    #     cx, cy = int(lm.x * w), int(lm.y * h)
    #     cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)

# This is the code for the image resize and display
original_shape = img.shape[:2]

# Calculate the scaling factor based on the desired window size
desired_width = 400  # Desired window width
desired_height = int(desired_width * original_shape[0] / original_shape[1])
# Desired window height, maintaining aspect ratio

# Set the window size and display the image
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', desired_width, desired_height)
cv2.imshow('Image', img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
