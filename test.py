import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

#Load the "mnist_test.csv" dataset.
data = pd.read_csv("mnist_test.csv")

#Separate the features and target
x = data.drop(columns=['label']).to_numpy()
y = data['label'].to_numpy()

#Normalize each image by dividing each pixel by 255.
x=x/255

# Load the model from the file
with open('saved_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

predictions = loaded_model.predict(x)

print("Predictions:",predictions)

accuracy = accuracy_score(y, predictions)
print("Accuracy for test data:",(accuracy*100), "%")

