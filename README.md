# Is-That-Santa-Image-Classification-using-CNN
This project implements a Convolutional Neural Network (CNN) to classify images as either containing Santa Claus or not. The focus is on building a deep learning model, training it effectively, and evaluating its performance using standard classification metrics.

Objective
To design and evaluate a CNN model capable of performing binary image classification for identifying whether an image depicts Santa Claus.

Dataset Overview
The dataset consists of labeled images divided into two categories: Santa and Not Santa

Separate folders were used for training and testing

Images were resized to 64x64 pixels and normalized before training

Model Architecture
A sequential CNN consisting of four convolutional layers with filter sizes of 32, 64, 128, and 512

Each convolutional layer is followed by:

ReLU activation

2x2 max pooling

Dropout for regularization

The final layers include:

Flattening layer

Dense layer with 1024 units (ReLU)

Dense layer with 512 units (ReLU)

Output layer with 1 unit and sigmoid activation for binary classification

Compilation and Training
The model was compiled twice:

First with the Adam optimizer for 15 epochs

Then with the SGD optimizer for 20 epochs

Loss function used: Binary Crossentropy

Evaluation metric: Accuracy

Performance Summary
Training Results
With Adam optimizer (15 epochs):

Training accuracy improved from 47.21% to 94.46%

Validation accuracy increased in parallel

With SGD optimizer (20 epochs):

Training accuracy stabilized at 95.62%

Validation accuracy remained between 93% and 95%

Additional training showed minimal improvement

Final Evaluation
Accuracy on test data: 51%

Classification Report:

Precision: 51%

Recall: 47.5%

F1-score: 49.6%

Confusion matrix revealed high false positives and false negatives

Key Observations
The model showed strong performance on training data but failed to generalize well on unseen data

Signs of overfitting were evident

Future improvements could include:

Regularization techniques such as early stopping or L2 penalty

Expanded and more diverse training data

Enhanced data augmentation strategies

Addressing class imbalance in the dataset

Technologies Used
Python

TensorFlow / Keras

NumPy

Scikit-learn

Matplotlib and Seaborn
