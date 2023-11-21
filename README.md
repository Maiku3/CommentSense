# CommentSense
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
## Overview
This project involves analyzing the sentiment expressed in comments on YouTube videos. The only sentiment indicator of the Youtube videos is the amount of likes as you can no longer view the amount of dislikes. Through the use of Machine Learning and Natural language processing, it is possible to analyze the sentiment of comments, which can provide more valuable insight into whether viewers like or dislike a video.

## Screenshots
![Screenshot 2023-11-13 120135](https://github.com/Maiku3/YT-Comment-Analysis/assets/95307563/32432cfa-351b-4d09-b9d7-7c1652a3f0fa)

## Design / Planning
![diagram](https://github.com/Maiku3/YT-Comment-Analysis/assets/95307563/2353fde5-c236-46d8-becc-67fc0f38e4d0)

## Model Accuracy
We got **80.25%** accuracy on the training set and **80.34%** on the validation set. The accuracy of the model is still **~80%** on the validation set, showing that our model generalized pretty well from the training data and is not overfitting.
![Screenshot 2023-11-13 131806](https://github.com/Maiku3/YT-Comment-Analysis/assets/95307563/679fbfdc-8845-4607-b348-4c78ea27415a)

## Dataset
First, to train a model, a dataset of text with sentiment labels is needed. This [dataset](https://www.kaggle.com/datasets/kazanova/sentiment140), Sentiment140 from Kaggle is used to train the model. It is a dataset of 1.6 million tweets labeled as positive, neutral, or negative. It would have been ideal if I could find a dataset of YouTube comments as that is what is being analyzed and would better fix that context.

## How to set up and run the application: 
Make sure you have a C++ compiler installed on your system.
1. Clone the repository and make sure you are in that directory:
```
https://github.com/Maiku3/CommentSense.git
```
2. Install the required packages
```
$ pip install -r requirements.txt
```
3. Run the flask app
```
$ python -m flask --app app run
```
## To be Improved on
- Making a more complex model to have the most accurate predictions.
- Hosting the website on a server and improving the UI.
