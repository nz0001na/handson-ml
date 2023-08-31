from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_mldata
from sklearn import datasets
import os
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score

custom_data_home = 'datasets/example-3-1/MNIST/'
if os.path.exists(custom_data_home) is False:
    os.makedirs(custom_data_home)

train_X, train_y = loadlocal_mnist(
        images_path=custom_data_home + 'train-images-idx3-ubyte/train-images.idx3-ubyte',
        labels_path=custom_data_home + 'train-labels-idx1-ubyte/train-labels.idx1-ubyte')

test_X, test_y = loadlocal_mnist(
    images_path=custom_data_home + 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte',
    labels_path=custom_data_home + 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'
)

# # show one image
feature = train_X[36000]
la=train_y[36000]
img = feature.reshape(28,28)
plt.imshow(img, interpolation='nearest') #cmap = matplotlib.cm.binary,
plt.axis('off')
plt.show()

shuffle_index = np.random.permutation(train_y.shape[0])
train_X = train_X[shuffle_index]
train_y = train_y[shuffle_index]

# true for all 5 numbers, False for all other digits
train_y_5 = (train_y == 5)
test_y_5 = (test_y == 5)

# test
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(train_X, train_y_5)
predict_y = sgd_model.predict([feature])

# cross-validation
accuracy = cross_val_score(sgd_model, train_X, train_y_5, cv=3, scoring='accuracy')

train_y_predict = cross_val_predict(sgd_model,train_X, train_y_5, cv=3)
m = confusion_matrix(train_y_5,train_y_predict)
print(m)

score = sgd_model.decision_function([train_X[100]])
y_scores = cross_val_predict(sgd_model, train_X, train_y_5, cv=3, method='decision_function')


precisions, recalls, thresholds = precision_recall_curve(train_y_5, y_scores)
# precision and recall vs. thresholds
plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
plt.plot(thresholds, recalls[:-1], 'g-', label='recall')
plt.xlabel('thresholds')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()

# recall vs. precision
plt.plot(recalls[:-1], precisions[:-1], 'g-')
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()

# compute ROC
fpr, tpr, thresholds = roc_curve(train_y_5, y_scores)
plt.plot(fpr, tpr, 'g-')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

# AUC
auc_score = roc_auc_score(train_y_5, y_scores)



svm_model = SGDClassifier()
svm_model.fit(train_X, train_y)
predict_y = cross_val_predict(svm_model, test_X, test_y, cv=3)
con = confusion_matrix(test_y, predict_y)
plt.matshow(con)
plt.show()





