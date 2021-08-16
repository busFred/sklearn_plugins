#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn_plugins.rvm.rvr import RVR

#%%
N_SAMPLES: int = 5000

#%%
np.random.seed(0)
Xc = np.ones([N_SAMPLES, 1])
Xc[:, 0] = np.linspace(-5, 5, N_SAMPLES)
Yc = 10 * np.sinc(Xc[:, 0]) + np.random.normal(0, 1, N_SAMPLES)
X_train, X_val, y_train, y_val = train_test_split(Xc,
                                                  Yc,
                                                  test_size=0.5,
                                                  random_state=0)

#%%
rvr: RVR = RVR(max_iter=300, verbose=True)
rvr.fit(X=X_train, y=y_train)
#%%
pred_train, train_var = rvr.predict(X=X_train, return_var=True)
train_std = np.sqrt(train_var)
mse_train: float = mean_squared_error(y_true=y_train, y_pred=pred_train)
print(str.format("mse_train: {}", mse_train))
#%%
pred_val, val_var = rvr.predict(X=X_val, return_var=True)
val_std: np.ndarray = np.sqrt(val_var)
mse_val: float = mean_squared_error(y_true=y_val, y_pred=pred_train)
print(str.format("mse_val: {}", mse_val))
#%%
plt.plot()
plt.scatter(X_train, y_train)
plt.scatter(X_train, pred_train)
plt.scatter(X_train, pred_train+train_std)
plt.scatter(X_train, pred_train-train_std)
#%%
plt.plot()
plt.scatter(X_val, y_val)
plt.scatter(X_val, pred_val)
plt.scatter(X_val, pred_val+val_std)
plt.scatter(X_val, pred_val-val_std)