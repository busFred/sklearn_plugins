#%%
import numpy as np
from sklearn import datasets
from sklearn_plugins.rvm.rvr import RVR

#%%
data, target = datasets.load_diabetes(return_X_y=True)
X: np.ndarray = np.expand_dims(data[:, 2], axis=1)
y: np.ndarray = target

#%%
X_train: np.ndarray = X[:-20]
y_train: np.ndarray = y[:-20]
X_val: np.ndarray = X[-20:]
y_val: np.ndarray = y[-20:]

#%%
rvr: RVR = RVR(max_iter=10000, verbose=True)
rvr.fit(X=X_train, y=y_train)
# %%
y_train_pred: np.ndarray = rvr.predict(X_train)
# %%
print(np.average(np.abs(y_train_pred - y_train)))
# %%
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train)
plt.scatter(X_train, y_train_pred)
# %%
