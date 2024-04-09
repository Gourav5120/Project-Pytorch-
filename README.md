Project Report: Chest Disease Detection 
1. Introduction
In this project, we aimed to develop a deep learning model for image classification using PY Torch. We focused on classifying medical images into three categories: COVID, NORMAL, and PNEUMONIA. The project involved various stages, including data preprocessing, model architecture selection, training, evaluation, and analysis.

2. Dataset
2.1 Dataset Description
The dataset consists of medical images collected from various sources, including COVID-19 datasets and public repositories.
It comprises three classes: COVID, NORMAL, and PNEUMONIA.
The dataset is stored in Google Drive and mounted in Google Collab for easy access during training.
2.2 Data Preprocessing
Data preprocessing involved resizing images, normalizing pixel values, and splitting the dataset into training, validation, and test sets.
Augmentation techniques such as random rotation, flipping, and scaling were applied to increase the diversity of the training data.
3. Model Architecture
3.1 Model Selection
We selected the Res Net architecture as the base model due to its effectiveness in image classification tasks.
The pre-trained ResNet-18 model was used and fine-tuned for our dataset.
3.2 Architecture Details
The model consisted of convolutional layers followed by fully connected layers.
We replaced the last fully connected layer to adapt the model to our three-class classification task.
The model was trained using the Adam optimizer with a learning rate of 0.001.
 
4. Training Process
The model was trained using the training dataset for a specified number of epochs.
We monitored training progress, including loss and accuracy, and employed early stopping to prevent overfitting.
Hyperparameter tuning was performed to optimize model performance, including learning rate and batch size.
 
5. Evaluation Results
The trained model was evaluated using the validation dataset to assess its performance.
Evaluation metrics such as accuracy, precision, recall, and F1-score were calculated.
Visualizations, including the confusion matrix, precision-recall curve, and ROC curve, were utilized to analyze model performance.
Classification Report:
              precision    recall  f1-score   support

       COVID       0.95      0.97      0.96       163
      NORMAL       0.90      0.90      0.90       181
   PNEUMONIA       0.88      0.85      0.87       163

    accuracy                           0.91       507
   macro avg       0.91      0.91      0.91       507
weighted avg       0.91      0.91      0.91       507
6. Testing on Held-out Data
The final model was tested on the held-out test dataset to assess its generalization to unseen data.
Test metrics, including accuracy, precision, recall, and F1-score, were computed.
Visualizations, such as the confusion matrix, were used to visualize model performance on the test dataset.
7. Conclusion
The developed image classification model demonstrated promising performance on the validation and test datasets.
Insights gained from the evaluation process provide valuable information for further improvements and future research.
The project highlights the potential of deep learning techniques for medical image analysis tasks.

 
