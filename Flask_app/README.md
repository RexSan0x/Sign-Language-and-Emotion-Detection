# Flask Application for Emotion and Sign Language Detection

This folder contains the files required to run our flask application for emotion Identification from images and Sign Language detection from Real-time Video frames.

## File Structure

<pre><font color="#12488B"><b>.</b></font>
├── <font color="#26A269"><b>app_final.py</b></font>
├── <font color="#12488B"><b>models</b></font>
│   ├── <font color="#26A269"><b>emotion_det.h5</b></font>
│   ├── <font color="#26A269"><b>signLang_det.h5</b></font>
│   └── <font color="#26A269"><b>something.txt</b></font>
├── <font color="#26A269"><b>requirements.txt</b></font>
├── <font color="#12488B"><b>static</b></font>
│   ├── <font color="#12488B"><b>css</b></font>
│   │   ├── <font color="#26A269"><b>Emotion_fileUpload.css</b></font>
│   │   ├── <font color="#26A269"><b>Emotion_pred.css</b></font>
│   │   ├── <font color="#26A269"><b>homepage.css</b></font>
│   │   └── <font color="#26A269"><b>Sign_Language.css</b></font>
│   ├── <font color="#12488B"><b>Image_emotion</b></font>
│   │   ├── <font color="#26A269"><b>Temp_emotion_predImg.png</b></font>
│   │   └── <font color="#26A269"><b>Temp_img.png</b></font>
│   ├── <font color="#12488B"><b>scriptjs</b></font>
│   │   └── <font color="#26A269"><b>Emotion_fileUpload.js</b></font>
│   └── <font color="#12488B"><b>Sign_Language_Chart</b></font>
│       └── <font color="#26A269"><b>Sign_language_chart.png</b></font>
└── <font color="#12488B"><b>templates</b></font>
    ├── <font color="#26A269"><b>Emotion_fileUpload.html</b></font>
    ├── <font color="#26A269"><b>Emotion_pred.html</b></font>
    ├── <font color="#26A269"><b>homepage.html</b></font>
    └── <font color="#26A269"><b>Sign_Language.html</b></font>
</pre>

The folder contains the following Structure:
1. **app_final.py**: This file contains the flask implementation of the application.
2. **static folder**: This folder contains the following folders:
    * **css**: Contains all CSS files needed to style the application.
    * **Image_emotion**: Acts as a temporary storage for images uploaded by the user for emotion prediction and emotion-predicted image.
    * **scriptjs**: Contains the javascript files necessary to run this application (mainly for uploading of image by user).
3. **templates folder**: Contains all HTML files needed by the application.
4. **models**: Contains the models that our team trained, used by our application to detect Emotion and Sign Language from an image or Video frames.
5. **requirements.txt**: Text file that contains the necessary libraries to be installed to run the application.



## Setup

1. Python version used - `3.8.16`
2. Tensorflow version - `2.12.0`
3. All training was done on google colab.
4. Dataset used for:
    * **Emotion Detection**: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
    * **Sign Language Identification**: Made by ourselves (Refer to **Sign_Language_Training** Direcotry.


## Running the application

To run the application locally, follow these steps:
1. Clone the repository by running the command: <br> `git clone https://github.com/RexSan0x/Sign-Language-and-Emotion-Detection.git`
2. Change your current working directory to the cloned repository. Navigate to the **Flask_app** folder using the command: <br>`cd Sign-Language-and-Emotion-Detection/Flask_app/`
3. Install all the required libraries for the application by running the following command:<br> `pip install -r requirements.txt`
4. Start the application by executing the command:<br> `python app_final.py`
5. Once the application is running, copy the localhost link provided in the terminal output and paste it into your web browser.

By following these steps, you will be able to run the application locally on your machine.
