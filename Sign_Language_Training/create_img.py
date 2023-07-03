'''
Purpose of this code is to capture the images from the user's camera and use them as our dataset to train our classifier.
'''

import os
import time
import cv2

# Directory to save our data
DATA_DIR = './Data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes - 26 alphabets
# Dataset size - 400 (400 for each right and left hand)
number_of_classes = 26
dataset_size = 400

cap = cv2.VideoCapture(0)
# while(True):
#     ret, frame = cap.read()
#     # Naming a window
#     cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    
#     # Using resizeWindow()
#     cv2.resizeWindow("frame", 700, 700)
#     cv2.imshow('frame', frame)
#     cv2.waitKey(25)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    for kdb in range(2): # For right and left hand
        while True:
            ret, frame = cap.read()  # Capture frame
            if kdb == 0: # For right hand
                cv2.putText(frame, 'Ready? Press "Q" - right! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                            cv2.LINE_AA)
            else: # For left hand
                cv2.putText(frame, 'Ready? Press "Q" - left! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                            cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):  # Quit the while loop
                break

        
        counter = 0
        while counter < dataset_size:  # Capture 400 frames from the users camera
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(dataset_size*kdb + counter)), frame)

            counter += 1

cap.release()  # Release capture object
cv2.destroyAllWindows()  # Destroy all windows
