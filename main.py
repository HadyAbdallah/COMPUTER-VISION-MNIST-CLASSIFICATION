import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

#Load the "mnist_train.csv" dataset.
data = pd.read_csv("mnist_train.csv")

print(data)
print(data.dtypes)
print("data set shape:", data.shape)
print('----------------------')

x = data.drop(columns=['label'])
y = data['label']


num_classes = y.nunique()
print("Number of unique classes:",num_classes)
print('----------------------')

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
k_range = list(range(1, 20, 2))  # Odd values from 1 to 30
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


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)