import cv2
import mediapipe as mp
import math
import urllib.request


def download_image(url, filename):
    url = urllib.request.urlopen(url)
    with open(filename, 'wb') as file:
        file.write(url.read())


# Usage
image_url = ('https://media.istockphoto.com/id/943001012/photo/front-view-full-portrait-of-young-charming-calm-lady'
             '-with-rolled-up-sleeves-standing-against.jpg?s=2048x2048&w=is&k=20&c=SpnldQm3'
             '-lfX68K8pPWyN6sQlkGt8ZFfNFqQCAZZqcU=')
save_as = 'image.jpg'

download_image(image_url, save_as)


# variables used for diff purpose in pose detection
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Define the threshold angle for correct posture
CORRECT_POSTURE_THRESHOLD = {
    'shoulder': 130,
    'hip': 30,
    'knee': 10,
    'elbow': 30
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
    right_elbow_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER],
                                        pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW],
                                        pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST])

    left_elbow_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER],
                                       pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW],
                                       pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST])

    # Calculate the angle between the shoulders and hips
    right_shoulder_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW],
                                           pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER],
                                           pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP])

    left_shoulder_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW],
                                          pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER],
                                          pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP])

    # Calculate the angle between the hips and knees
    right_hip_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER],
                                      pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP],
                                      pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE])
    left_hip_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER],
                                     pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP],
                                     pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE])

    # Calculate the angle between the left and right knees and ankles
    right_knee_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP],
                                       pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE],
                                       pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ANKLE])

    left_knee_angle = calculate_angle(pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP],
                                      pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE],
                                      pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ANKLE])

    # Check if the posture is correct
    is_posture_correct = all([left_shoulder_angle >= CORRECT_POSTURE_THRESHOLD['shoulder'],
                              right_shoulder_angle >= CORRECT_POSTURE_THRESHOLD['shoulder'],
                              left_elbow_angle <= CORRECT_POSTURE_THRESHOLD['elbow'],
                              right_elbow_angle <= CORRECT_POSTURE_THRESHOLD['elbow'],
                              right_hip_angle <= CORRECT_POSTURE_THRESHOLD['hip'],
                              left_hip_angle <= CORRECT_POSTURE_THRESHOLD['hip'],
                              right_knee_angle <= CORRECT_POSTURE_THRESHOLD['knee'],
                              left_knee_angle <= CORRECT_POSTURE_THRESHOLD['knee']])
    print(left_elbow_angle, "Left Elbow Angle ")
    print(right_elbow_angle, "Right Elbow Angle ")
    print(left_shoulder_angle, "Left Shoulder Angle ")
    print(right_shoulder_angle, "Right Shoulder Angle ")
    print(left_hip_angle, "Left Hip Angle ")
    print(right_hip_angle, "Right Hip Angle ")
    print(left_knee_angle, "left Knee Angle ")
    print(right_knee_angle, "Right Knee Angle ")
    if is_posture_correct:
        print("Posture is correct")
    else:
        print("Posture is incorrect")


# it reads the image file and convert it to RGB from BGR and process the landmarks and draw the landmarks on the image
img = cv2.imread(save_as, 1)

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
desired_width = 500  # Desired window width
desired_height = int(desired_width * original_shape[0] / original_shape[1])
# Desired window height, maintaining aspect ratio

# Set the window size and display the image
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', desired_width, desired_height)
cv2.imshow('Image', img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
