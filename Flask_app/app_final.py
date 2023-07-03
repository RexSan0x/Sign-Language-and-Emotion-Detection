from flask import Flask,render_template,Response, url_for, redirect, request
import cv2
import mediapipe as mp
import numpy as np
import pickle
from keras.models import  load_model

app=Flask(__name__)
cap = ""  # Video Capture varriable of cv2

def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)
    #global temp_text

    ##### Load your model here
    model_dict = pickle.load(open('models/signLang_det.p', 'rb'))  ##Original model model2.p
    model = model_dict['model']

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {}
    for i in range(26):
        labels_dict[i] = chr(65+i)
        
    while True:

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        try:
            H, W, _ = frame.shape
        except AttributeError:
            print("Frame not displaying anymore!!")

        temp_frame = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        # if not results.multi_hand_landmarks:
        #     temp_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            try:

                ###### Change this with your model

                prediction = model.predict([np.asarray(data_aux)])
                #print(f"Predictions: {prediction}")

                predicted_character = labels_dict[int(prediction[0])]
                # temp_text = predicted_character

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            except ValueError:
                frame = temp_frame
                print("Value Error occured!!")
                #temp_text = ""



        # print(temp_text)    
        
        # # Writing to sample.json
        # with open("sample.json", "w") as outfile:
        #     outfile.write(json.dumps({"Letter":temp_text}, indent=4))
            
        # ## read the camera frame
        # success,frame=camera.read()
        # if not success:
        #     break
        # else:
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


## Homepage
@app.route('/')
def index():
    return render_template('homepage.html')

## Redirect to homepage
@app.route('/Start')
def Start():
    global cap
    if cap != "":
        cap.release()
    return render_template('homepage.html')

## Go to Sign Language
@app.route('/Sign')
def Sign():
    return render_template('Sign_Language.html')

## Generate video frame from camera and print it to screen
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

## Emotion Detection
@app.route('/emotion')
def emotion():
    return render_template('Emotion_fileUpload.html')

## Get POST request with file/image to detect emotion
@app.route('/form', methods=['POST'])
def upload_file():

    # If 'imageFile' (file tag) present in request (If file is present)
    if 'imageFile' in request.files:
        image_file = request.files['imageFile']  # Check image from request
        file_name = image_file.filename  #Filename
        print(f"Uploaded file name: {file_name}")

        # Save image as 'Temp_img.png'
        img_path = 'static/Image_emotion/Temp_img.png'
        image_file.save(img_path)
    else:
        print("No file uploaded")

    # Read file from the image path set
    test_img = cv2.imread(img_path)

    # Converting from BGR -> RGB
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # Loading the model
    model = load_model("models/emotion_det.h5")

    # Detecting face from the image
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    print("Faces: ", faces_detected)

    if len(faces_detected) == 0:   ##If no face detected

        # Data Pre-processing
        roi_gray = cv2.resize(gray_img, (48, 48))
        img_pixels = np.expand_dims(roi_gray, axis=0)
        img_pixels = img_pixels.astype(np.float32)  # Convert to float32
        img_pixels /= 255

        # Making predictions
        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise")
        predicted_emotion = emotions[max_index]
        resized_img = cv2.resize(test_img, (1000, 700))

    else:  ##If face has been detected
        print("Face outline Detected")

        # For coordinates of the face bounding box present in image
        for (x, y, w, h) in faces_detected:
            
            # Data Pre-processing
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = np.expand_dims(roi_gray, axis=0)
            img_pixels = img_pixels.astype(np.float32)  # Convert to float32
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise")
            predicted_emotion = emotions[max_index]

            # Calculate coordinates of a point from the original image in the resized image
            def calculate_resized_coordinates(original_width, original_height, resized_width, resized_height, point):
                # Calculate the ratio of width and height
                width_ratio = resized_width / original_width
                height_ratio = resized_height / original_height

                # Calculate the corresponding coordinates in the resized image
                resized_x = int(point[0] * width_ratio)
                resized_y = int(point[1] * height_ratio)

                return resized_x, resized_y

            # Calculating resized Coordinates
            x1, y1 = calculate_resized_coordinates(test_img.shape[1], test_img.shape[0], 1000, 700, (int(x), int(y)))
            x2, y2 = calculate_resized_coordinates(test_img.shape[1], test_img.shape[0], 1000, 700, (int(x+w), int(y+h)))

            # Resize image
            resized_img = cv2.resize(test_img, (1000, 700))

            # Draw retangle and put text
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=4)  
            cv2.putText(resized_img, predicted_emotion, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Saving image to path with filename - Temp_emotion_predImg.png
    cv2.imwrite('static/Image_emotion/Temp_emotion_predImg.png', resized_img)
    print("Prediction: ", predicted_emotion)

    # Redirect to prediction page with information - predicted_emotion
    return render_template("Emotion_Pred.html", pred = predicted_emotion)

# Redirect back to Uploading file for emotion prediction Page
@app.route('/back')
def back():
    return render_template('Emotion_fileUpload.html')

if __name__=="__main__":
    app.run()
