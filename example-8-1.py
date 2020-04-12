from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import numpy as np

mnist = fetch_mldata('MNIST original')

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

n_batches = 100
ica = IncrementalPCA(n_components=154)
for n_batch in np.array_split(X_train, n_batches):
    ica.partial_fit(n_batch)
    print('d')
X1 = ica.transform(X_train)
print(X1)

X0 = ica.inverse_transform(X1)