import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#Load the "mnist_train.csv" dataset and perform initial data exploration..
data = pd.read_csv("mnist_train.csv")

print(data)
print('----------------------')
print(data.dtypes)
print('----------------------')
print("data set shape:", data.shape)
print('----------------------')

#Separate the features and target
x = data.drop(columns=['label'])
y = data['label']

#Identify the number of unique classes.
num_classes = y.nunique()
print("Number of unique classes:",num_classes)
print('----------------------')

#Identify the number of features.
num_features = len(x.columns)
print("Number of features (pixels):", num_features)
print('----------------------')

# Check whether there are missing values
missing_values = data.isnull().sum()
print('Missing values:')
print(missing_values)
print('----------------------')

#Normalize each image by dividing each pixel by 255.
x=x/255


'''
Resize images to dimensions of 28 by 28.
After resizing, visualize some images to verify the correctness of the reshaping process.
'''
resized_image=[]
for i in range(len(x)):
    img = np.array(x.loc[i]).reshape(28, 28)
    resized_img = resize(img, (28, 28))
    resized_image.append(resized_img.flatten())

fig, axes = plt.subplots(1,4 , figsize=(10, 3))
for i in range(4):
    img = resized_image[i].reshape(28, 28)
    axes[i].imshow(img)
    axes[i].axis('off')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Create a k-NN classifier
knn = KNeighborsClassifier()


# define the parameter values that should be searched
k_range = list(range(1, 20, 2))  # Odd values from 1 to 20
weight_options = ['uniform', 'distance']
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)



# instantiate the grid
grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1)



grid_search.fit(x_train, y_train)



# view the results
pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]



best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)



best_knn = KNeighborsClassifier(**best_params)
best_knn.fit(x_train, y_train)



y_pred = best_knn.predict(x_test)


best_knn_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", (best_knn_accuracy*100), "%")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


##########################################
#Traning a Neural network
from sklearn.neural_network import MLPClassifier

# Define the first ANN architecture
ann1 = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    random_state=100
)

# Train the first ANN
ann1.fit(x_train, y_train)

# Evaluate the first ANN on the validation set
y_pred_ann1 = ann1.predict(x_test)

accuracy_ann1 = accuracy_score(y_test, y_pred_ann1)
print("Accuracy (ANN1):", (accuracy_ann1*100), "%")

# Define the second ANN architecture with different hyperparameters
ann2 = MLPClassifier(
    hidden_layer_sizes=(50,),    # Single hidden layer with 50 neurons
    learning_rate_init=0.01,      # Initial learning rate
    batch_size=128,               # Batch size
    max_iter=500,                 # Maximum number of iterations
    random_state=100
)

# Train the second ANN
ann2.fit(x_train, y_train)

# Evaluate the second ANN on the validation set
y_pred_ann2 = ann2.predict(x_test)

accuracy_ann2 = accuracy_score(y_test, y_pred_ann2)
print("Accuracy (ANN2):", (accuracy_ann2*100), "%")

# Choose the best model based on validation accuracy
best_ann,best_ann_accuracy = (ann1,accuracy_ann1) if accuracy_ann1 >= accuracy_ann2 else (ann2,accuracy_ann2)

print("Best ANN architecture:", "ANN1" if accuracy_ann1 >= accuracy_ann2 else "ANN2")


# Plot the performance comparison as a horizontal bar plot with a smaller y-axis scale
labels = ['ANN1', 'ANN2']
accuracies = [accuracy_ann1, accuracy_ann2]

plt.figure(figsize=(8, 3))  # Smaller height for the plot
plt.barh(labels, accuracies, color=['blue', 'orange'])
plt.xlabel('Accuracy')
plt.title('Performance Comparison between ANN1 and ANN2')
plt.xlim(0.94, 0.99)  # Adjusted x-axis limit to focus on differences
plt.show()

# Choose the best model based on validation accuracy
best_model,best_accuracy = (best_ann,best_ann_accuracy) if best_ann_accuracy >= best_knn_accuracy else (best_knn,best_knn_accuracy)
print("Best Mode architecture:", "ANN" if best_ann_accuracy >= best_knn_accuracy else "K-NN")

#Confusion matrix of the best model
predictions = best_model.predict(x_test)
confusion = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(confusion, display_labels=[str(i) for i in range(10)]).plot(cmap='viridis')

#Save the best model
import pickle

with open('saved_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)