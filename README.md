# Project Name : Impact of Artificial Intelligence on Emotional Perception and Self-Management

# Table of Content :
1.[Description](#p1)

2.[Installations](#p2)

3.[Usage](#p3)

4.[Dataset](#p4)

<a id="p1"></a> 
# Description:

This study explores the impact of Artificial Intelligence on emotional perception and self-management by integrating AI-based emotion recognition with diverse visual feedback techniques such as colors, shapes, and text prompts. The research examines how these technologies collectively influence participants' awareness of their emotions and assesses the accuracy of the feedback, user acceptance, and its effects on emotional management capabilities. 

## What does Emotion Recognition mean?

Emotion recognition is a technique used in software that allows a program to "read" the emotions on a human face using advanced image processing. Companies have been experimenting with combining sophisticated algorithms with image processing techniques that have emerged in the past ten years to understand more about what an image or a video of a person's face tells us about how he/she is feeling and not just that but also showing the probabilities of mixed emotions a face could has.

<a id="p2"></a> 
# Installations:

Mac Install using:

Set up a new Python virtual environment：
```shell
conda init
```
```shell
conda create -n emotion-detect python=3.8 -y
```
```shell
conda activate emotion-detect
```

Installation library：
```shell
pip install opencv-python
pip3 install keras
pip3 install tensorflow
pip3 install imutils
```

<a id="p3"></a> 
# Usage:

The program will create a window to display the scene captured by the webcam and a window to represent the probability of the detected emotion. Depending on the label of the identified emotion and the percentage of the emotion, a visual feedback (shape, colour, textual cues) corresponding to the emotion appears. You can open the ‘experiment’ folder for different visualizations of experiment analogies. You can also put video in to run.

> Demo

python real_time_final.py

You can just use this with the provided pretrained model i have included in the path written in the code file, i have choosen this specificaly since it scores the best accuracy, feel free to choose any but in this case you have to run the later file train_emotion_classifier
> If you just want to run this demo, the following content can be skipped
- Train

- python train_emotion_classifier.py


<a id="p4"></a> 
# Dataset:

I have used [this](https://www.kaggle.com/c/3364/download-all) dataset

Download it and put the csv in fer2013/fer2013/

-fer2013 emotion classification test accuracy: 66%

and the private facial data I collected  collected by recruiting volunteers.


# Credits
This work is inspired from [this](https://github.com/otaha178/Emotion-recognition.git) great work and the resources of Emotion recognition helped me alot!

# Ongoing 
Draw emotions faces next to the detected face.
