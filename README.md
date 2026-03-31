# Credit Score Classification Project / Artificial Neural Network (ANN)



## Overview



This project focuses on evaluating credit risk using a dataset containing 50,000 records from 12,500 customers. The dataset provides a comprehensive view of customer profiles, including demographic information, financial histories, and payment patterns. The primary goal is to clean and analyze this data to identify key features and train Deep Learning algorithms for credit score prediction. The target variable, Credit Score, is categorized into three classes: Standard, Poor, and Good.


### Data Preprocessing & Exploratory Data Analysis (EDA)



**Data Cleaning**: Hexadecimal IDs were converted to decimal formats. Categorical and string features such as Month, Occupation, and Credit Mix were encoded into numeric values. Additionally, the credit history age was recalculated into total months.


**Scaling**: Data was scaled between 0 and 1 to optimize the Artificial Neural Network (ANN) training process, as neural networks are highly sensitive to the scale of inputs.


**EDA Insights**: Correlation analysis revealed a perfect 1.0 correlation between annual income and monthly in-hand salary. A strong positive correlation (0.81) was also found between income and monthly investment amounts, perfectly aligning with real-world financial principles. Furthermore, analysis showed that customers with a "Bad" credit mix consistently had significantly higher interest rates and outstanding debts.


### Model Architecture (Artificial Neural Network)



A Deep Learning model was developed with 4 hidden layers, initially expanding to 512 neurons and then narrowing down to 256 to extract and distill the purest features.


**Activations**: ReLU was used in hidden layers for faster learning, while Softmax was used in the output layer to calculate the probability of the 3 credit score classes.


**Optimization & Regularization**: To prevent overfitting, a 35% Dropout rate and L1 Regularization were implemented. The Adam optimizer (with a learning rate of 0.0003) and Early Stopping criteria (35 epochs) were utilized for precise learning.


**Imbalanced Data Handling**: Class Weights were applied during training to assign higher penalty costs to the minority classes, ensuring balanced learning.


### Results & Conclusion



**Performance**: The model achieved a 100% success rate and a 1.0 ROC AUC score on the test set.


**Synthetic Data Discovery**: These exceptionally high validation metrics indicated that the dataset was synthetic. The ANN model successfully deciphered the underlying, hidden mathematical formula used to generate the data.


**Dummy Customer Validation**: To validate the model's actual logic, it was tested on 50 randomly generated, unseen "dummy" customers. The model correctly penalized high-debt profiles by immediately classifying them as "Poor" and rewarded low-debt/low-interest profiles as "Good". This test proved that the model did not merely memorize the numbers, but genuinely learned the fundamental financial logic: "higher debt and interest rates indicate higher risk".
