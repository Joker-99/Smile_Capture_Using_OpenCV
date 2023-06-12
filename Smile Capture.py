from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import dlib
import cv2


# Calculate distances between specific landmarks of the mouth

def smile(mouth):
    
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A + B + C) / 3

    # Calculate the width of the mouth by finding the distance between the corners
    D = dist.euclidean(mouth[0], mouth[6])

    # Calculate the mouth aspect ratio (MAR)
    mar = avg / D
    return mar

# Initialize counters for detecting drowsiness
COUNTER = 0
TOTAL = 0

shape_predictor = "shape_predictor_68_face_landmarks.dat"  # Path to facial landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] Starting to Capture...")
# Start the video stream from the default camera
vs = VideoStream(src=0).start()  
fileStream = False
time.sleep(1.0)

# Initialize a FPS (Frames Per Second) counter
Cheese = FPS().start()  
cv2.namedWindow("SMILE!")

while True:
    frame = vs.read()  # Read a frame from the video stream
    frame = imutils.resize(frame, width=450)  # Resize the frame to a smaller width for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    rects = detector(gray, 0)  # Detect faces in the grayscale frame

    for rect in rects:
        shape = predictor(gray, rect)  # Predict the facial landmarks for each detected face
        shape = face_utils.shape_to_np(shape)  # Convert the predicted landmarks to NumPy array format
        mouth = shape[mStart:mEnd]  # Extract the mouth landmarks from the predicted landmarks
        mar = smile(mouth)  # Calculate the mouth aspect ratio (MAR)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)  # Draw the mouth contour on the frame

        if mar <= 0.3 or mar > 0.38:
            # If the mouth aspect ratio falls below a threshold or exceeds another threshold, increment the counter
            COUNTER += 1
        else:
            if COUNTER >= 15:
                # If the counter exceeds a certain value, it means the person is drowsy, so capture a frame
                TOTAL += 1
                frame = vs.read()
                time.sleep(0.3)
                frame2 = frame.copy()
                img_name = "opencv_frame_{}.png".format(TOTAL)
                cv2.imwrite(img_name, frame2)  # Save the captured frame as an image file
                print("{} written!".format(img_name))
            COUNTER = 0

        cv2.putText(frame, "Mouth Aspect Ratio: {:.2f}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)  # Display the mouth aspect ratio on the frame

    cv2.imshow("Cheese!", frame)  # Display the frame
    Cheese.update()  # Update the FPS counter

    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):  # If 'q' is pressed, exit the loop
        break

Cheese.stop()

cv2.destroyAllWindows()  # Close all windows
vs.stop()  # Stop the video stream
