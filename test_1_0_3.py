import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

mnist_train = pd.read_csv("mnist_train.csv", header=None)
mnist_test = pd.read_csv("mnist_test.csv", header=None)
#print(mnist_train)
cols = ["label"]
# for i in range(784):
#     cols.append(f"px_{i + 1}") # python = 3.6
for i in range(784):
    cols.append("px_" + str(i + 1))
# cols
mnist_train.columns = cols
mnist_test.columns = cols
mnist_train.head(7)
mnist_test.head(5)
image_size = 28 # размерность картинки
train_label = mnist_train["label"].values
print(train_label)
test_label = mnist_test["label"].values
print(test_label)
train_images = mnist_train.values[:, 1:]
test_images = mnist_test.values[:, 1:]
train_images = train_images.reshape(60000, image_size, image_size)
test_images = test_images.reshape(10000, image_size, image_size)
knn_classifier = KNeighborsClassifier(n_jobs=-1)
# fit/predict
#knn_classifier = knn_classifier.fit(train_images, train_label)
plt.imshow(train_images[432, :, :], cmap="Greys")

knn_classifier = knn_classifier.fit(train_images.reshape(60000, 784), train_label)
print(knn_classifier)
print("__________________")
image_id = 1
#plt.imshow(test_images[image_id], cmap="Greys")
#plt.show()

prediction = knn_classifier.predict(test_images[image_id].reshape(1, 784))
print(prediction)
plt.imshow(test_images[image_id], cmap="Greys")
#plt.show()
#get_ipython().run_line_magic('pinfo', 'accuracy_score')
# knn_classifier.predict
all_predictions = knn_classifier.predict(test_images.reshape(10000, 784))
print(accuracy_score(test_label, all_predictions) * 100)
cmPr = confusion_matrix(test_label, all_predictions)
print("точность предсказания" , cmPr)
#for i, (real, pred) in enumerate(zip(test_label, all_predictions)):
 ##      print("Prediction: " + str(pred))
   #     plt.imshow(test_images[i], cmap="Greys")
    #    plt.show()
#list(zip([1, 2, 3], ["I", "II", "III"]))
#list(enumerate(["a", "b", "c", "d", "e"]))
# Домашнее задание
# knn_classifier = KNeighborsClassifier(n_neighbors=???, n_jobs=-1)
# Попробовать другой алгоритм 

nn_classifier = MLPClassifier()
#get_ipython().run_line_magic('pinfo', 'nn_classifier.fit')
nn_classifier = nn_classifier.fit(train_images.reshape(60000, 784), train_label)
image_id = 123
#prediction = nn_classifier.predict(test_images[image_id].reshape(1, 784))
#print(prediction)
#plt.imshow(test_images[image_id], cmap="Greys")
all_predictions = nn_classifier.predict(test_images.reshape(10000, 784))
print(accuracy_score(test_label, all_predictions) * 100)
cmNet = confusion_matrix(test_label, all_predictions)
print("точность сети" , cmNet)
print(cmPr-cmNet)
#for i, (real, pred) in enumerate(zip(test_label, all_predictions)):
 #   if real == 4 and real != pred:
  #      print("Prediction: " + str(pred))
   ##    plt.show()
# Домашнее задание (часть 2)
# MLPClassifier(hidden_layer_sizes=???, activation=???)