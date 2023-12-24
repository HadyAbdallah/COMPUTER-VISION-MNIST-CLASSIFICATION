import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import resize


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