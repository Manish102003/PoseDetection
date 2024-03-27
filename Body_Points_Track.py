import cv2
import mediapipe as mp

# variables used for diff purpose in pose detection

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


# it reads the image file and convert it to RGB from BGR and process the landmarks and draw the landmarks on the image
img = cv2.imread("images/image6.jpg", 1)

if img is None:
    print("Error loading the image.")
    exit(1)

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = pose.process(imgRGB)

# print(result.pose_landmarks)

if result.pose_landmarks:
    mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id,lm in enumerate(result.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img,(cx, cy), 1, (255, 0, 0), cv2.FILLED)

# This is the code for the image resize and display
original_shape = img.shape[:2]
desired_width = 500
desired_height = int(desired_width * original_shape[0] / original_shape[1])

# Set the window size and display the image
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', desired_width, desired_height)
cv2.imshow('Image', img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.imshow("Image", img)
# cv2.waitKey(1000000)
