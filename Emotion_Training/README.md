## Emotion Detection

The emotion detection model utilizes a Convolutional Neural Network (CNN) architecture to classify facial expressions into several emotion categories. The model is trained on a diverse dataset of facial images representing emotions such as Anger, Disgust, Fear, Happy, Neutral, Sadness, and Surprise. Given an input image, the model extracts the face region using a Haar cascade classifier and predicts the corresponding emotion using the trained CNN model.

## Training

Training of the **Emotion Detection** model was done in [Google colab](https://colab.research.google.com/). Refer to `emotion.ipynb`

## Model Summary

<pre>
Model: "DCNN"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_1 (Conv2D)           (None, 48, 48, 64)        4864      
                                                                 
 batchnorm_1 (BatchNormaliza  (None, 48, 48, 64)       256       
 tion)                                                           
                                                                 
 conv2d_2 (Conv2D)           (None, 48, 48, 64)        102464    
                                                                 
 batchnorm_2 (BatchNormaliza  (None, 48, 48, 64)       256       
 tion)                                                           
                                                                 
 maxpool2d_1 (MaxPooling2D)  (None, 24, 24, 64)        0         
                                                                 
 dropout_1 (Dropout)         (None, 24, 24, 64)        0         
                                                                 
 conv2d_3 (Conv2D)           (None, 24, 24, 128)       73856     
                                                                 
 batchnorm_3 (BatchNormaliza  (None, 24, 24, 128)      512       
 tion)                                                           
                                                                 
 conv2d_4 (Conv2D)           (None, 24, 24, 128)       147584    
                                                                 
 batchnorm_4 (BatchNormaliza  (None, 24, 24, 128)      512       
 tion)                                                           
                                                                 
 maxpool2d_2 (MaxPooling2D)  (None, 12, 12, 128)       0         
                                                                 
 dropout_2 (Dropout)         (None, 12, 12, 128)       0         
                                                                 
 conv2d_5 (Conv2D)           (None, 12, 12, 256)       295168    
                                                                 
 batchnorm_5 (BatchNormaliza  (None, 12, 12, 256)      1024      
 tion)                                                           
                                                                 
 conv2d_6 (Conv2D)           (None, 12, 12, 256)       590080    
                                                                 
 batchnorm_6 (BatchNormaliza  (None, 12, 12, 256)      1024      
 tion)                                                           
                                                                 
 maxpool2d_3 (MaxPooling2D)  (None, 6, 6, 256)         0         
                                                                 
 dropout_3 (Dropout)         (None, 6, 6, 256)         0         
                                                                 
 flatten (Flatten)           (None, 9216)              0         
                                                                 
 dense_1 (Dense)             (None, 128)               1179776   
                                                                 
 batchnorm_7 (BatchNormaliza  (None, 128)              512       
 tion)                                                           
                                                                 
 dropout_4 (Dropout)         (None, 128)               0         
                                                                 
 out_layer (Dense)           (None, 7)                 903       
                                                                 
=================================================================
Total params: 2,398,791
Trainable params: 2,396,743
Non-trainable params: 2,048
_________________________________________________________________
</pre>

## Dataset

Google Drive Link: https://drive.google.com/drive/folders/1cbjNzbBKTqGbL8TD0J2SpdbqYuaXZokC?usp=drive_link

## Pre-trained model

Google Dirve link: https://drive.google.com/file/d/1slk39TbYdgLuRshw3nmeFiHm42S8mpwp/view?usp=drive_link
