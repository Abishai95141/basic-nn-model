# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network is a computational model inspired by the structure and function of the human brain. It is a type of machine learning algorithm that processes information through interconnected nodes, known as neurons or artificial neurons. These neurons are organized into layers: an input layer, one or more hidden layers, and an output layer. In a neural network, each connection between neurons has an associated weight, and the network learns by adjusting these weights based on input data and desired output.

The model below has one input layer,The input layer receives the initial data or features for the neural network to process. Each node in this layer represents a specific feature of the input data.

There are 5 hidden layers in this model, making it a deep neural network. These hidden layers are responsible for learning complex patterns and representations from the input data. Each hidden layer consists of nodes, and the connections between these nodes have associated weights. These weights are adjusted during the learning process to enhance the network's ability to capture and generalize from the input data.

The output layer produces the final result or prediction. Each node in the output layer represents either a class (for classification tasks) or a continuous value (for regression tasks). The exact nature of the output depends on the specific requirements of the problem being addressed.

## Neural Network Model

![Screenshot 2024-02-28 071343](https://github.com/Abishai95141/basic-nn-model/assets/139335314/509fd398-f8db-46cd-b9c3-e76707851642)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: ABISHAI K C
### Register Number: 212223240002
```python

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('regression_model').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})

X = df[['input']].values
y = df[['output']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

AI_Brain = Sequential([
    Dense(units = 1, activation = 'relu', input_shape=[1]),
    Dense(units = 5, activation = 'relu'),
    Dense(units = 1)
])

AI_Brain.compile(optimizer= 'rmsprop', loss="mse")
AI_Brain.fit(X_train1,y_train,epochs=3000)
AI_Brain.summary()

loss_df = pd.DataFrame(AI_Brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
AI_Brain.evaluate(X_test1,y_test)
X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
AI_Brain.predict(X_n1_1)



```
## Dataset Information

Include screenshot of the dataset

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Abishai95141/basic-nn-model/assets/139335314/8c8ec8ad-5f24-49e4-8041-0883570f8a28)


### Test Data Root Mean Squared Error

![image](https://github.com/Abishai95141/basic-nn-model/assets/139335314/48c4fda6-7d2a-4b2b-9f3e-f0f71e9766ab)


### New Sample Data Prediction

Include your sample input and output here

## RESULT

Include your result here
