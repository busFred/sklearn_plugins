#%%
from typing import List
import numpy as np
import onnxruntime as rt
from onnx import ModelProto
from skl2onnx import to_onnx
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn_plugins.rvm.rvc import RVC

#%%
N_SAMPLES: int = 5000
CENTERS = [(-3, -3), (0, 0), (3, 3)]

#%%
X, y = make_blobs(n_samples=N_SAMPLES,
                  n_features=2,
                  cluster_std=1.0,
                  centers=CENTERS,
                  shuffle=False,
                  random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X,
                                                  y,
                                                  test_size=0.5,
                                                  random_state=0)

#%%
rvc: RVC = RVC()
rvc.fit(X_train, y_train)

#%%
"""
Both float and double works with skl2onnx 1.9.1.dev
"""
onx: ModelProto
onx = to_onnx(rvc, X_train[:1, :].astype(np.float64), target_opset=13)

#%%
with open("rvc.onnx", "wb") as file:
    file.write(onx.SerializeToString())
#%%
sess = rt.InferenceSession(onx.SerializeToString())

#%%
pred_train: np.ndarray = rvc.predict(X_train)
prob_train: np.ndarray = rvc.predict_proba(X_train)
acc_train: float = accuracy_score(y_true=y_train, y_pred=pred_train)

#%%
results: List[np.ndarray] = sess.run(None, {'X': X_train.astype(np.float64)})
onnx_pred_train: np.ndarray = results[0]
#%%
print(np.sum(np.abs(pred_train-results[0])))
print(np.sum(np.abs(prob_train-results[1])))


#%%
pred_val: np.ndarray = rvc.predict(X_val)
prob_val: np.ndarray = rvc.predict_proba(X_val)
acc_val: float = accuracy_score(y_true=y_val, y_pred=pred_val)

#%%
print(str.format("train_acc {} val_acc {}", acc_train, acc_val))
