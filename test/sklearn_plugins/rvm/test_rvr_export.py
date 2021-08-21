#%%
from typing import List

import numpy as np
import onnxruntime as rt
from onnx import ModelProto
from skl2onnx.convert import to_onnx
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
rvr: RVR = RVR(max_iter=100, verbose=True, update_y_var=True)
rvr.fit(X=X_train, y=y_train)

#%%
"""
Both float and double works with skl2onnx 1.9.1.dev
"""
onx: ModelProto
onx = to_onnx(rvr, X_train[:1,:].astype(np.float64), target_opset=13)

#%%
with open("rvr.onnx", "wb") as file:
    file.write(onx.SerializeToString())

#%%
sess = rt.InferenceSession(onx.SerializeToString())
results: List[np.ndarray] = sess.run(None, {'X': X_train.astype(np.float64)})
onnx_pred_train: np.ndarray = results[0]

#%%
pred_train = rvr.predict(X=X_train)
mse_train: float = mean_squared_error(y_true=y_train, y_pred=pred_train)
print(str.format("mse_train: {}", mse_train))
print(
    str.format("pred_train - onnx_pred = {}",
               np.sum(np.abs(pred_train - onnx_pred_train))))

#%%
results: List[np.ndarray] = sess.run(None, {'X': X_val.astype(np.float64)})
onnx_pred_val: np.ndarray = results[0]
#%%
pred_val = rvr.predict(X=X_val)
mse_val: float = mean_squared_error(y_true=y_val, y_pred=pred_val)
print(str.format("mse_val: {}", mse_val))
print(
    str.format("pred_val - onnx_pred = {}",
               np.sum(np.abs(pred_val - onnx_pred_val))))

#%%
test = X_val[0:1, :]
print(test)
results: List[np.ndarray] = sess.run(None, {'X': test.astype(np.float64)})
onnx_pred_single: np.ndarray = results[0]
print(onnx_pred_val)
pred_single: np.ndarray = rvr.predict(test)
print(pred_single)

# %%
