# ANN-AUTOMOTIVE-VEHICLES-ENGINE-HEALTH-PREDICTION
developed a predictive maintenance model using a neural network to assess engine health. The model analyzes features like RPM and temperatures to predict maintenance needs. Evaluated with 66.65% accuracy, it helps optimize schedules and improve vehicle reliability, enhancing safety and cost savings.

# Introduction:

This project focuses on developing a predictive maintenance model for automotive engines using supervised learning. We utilize a feedforward neural network with a multi-layer perceptron (MLP) architecture, trained via backpropagation. The dataset includes features like engine RPM, lubricating oil pressure, fuel pressure, coolant pressure, lubricating oil temperature, and coolant temperature, with engine condition labels (1 for maintenance needed, 0 otherwise). Our objective is to predict engine condition using these features, facilitating proactive maintenance.

# Data and Preprocessing:

The dataset, sourced from Kaggle, contains 19,535 rows and 7 columns. We preprocess the data by normalizing and scaling features to a 0-1 range, ensuring optimal model performance. Outlier detection and removal were performed to eliminate anomalies. The dataset is split into 70% training and 30% testing subsets for model evaluation.

# Model Development:

We constructed an MLP model using Keras in R, with an input layer, three hidden layers, and an output layer. The model is compiled with the Adam optimizer and mean absolute error as the loss function. ReLU activation is used in hidden layers to introduce non-linearity, while a sigmoid function in the output layer maps predictions to probabilities. Training involved iteratively adjusting weights through backpropagation.

# Evaluation and Results:

The model's performance is evaluated using metrics like accuracy (66.65%), precision, recall (83.41%), and F1-score (76.18%). The confusion matrix provides insights into true positives, false positives, true negatives, and false negatives, allowing for a comprehensive performance assessment. Despite moderate accuracy, the model demonstrates effective engine failure detection, offering valuable insights for proactive maintenance planning.
![image](https://github.com/user-attachments/assets/f4d3e769-bf6c-4d9d-a277-447e9ab61508)
![image](https://github.com/user-attachments/assets/c46f58f1-137b-4fed-87d4-f181bd365369)


# Conclusion:

Our predictive maintenance model aids in identifying potential engine issues early, optimizing maintenance schedules, and enhancing vehicle performance. This approach can lead to cost savings, improved safety, and increased reliability in the automotive industry.
