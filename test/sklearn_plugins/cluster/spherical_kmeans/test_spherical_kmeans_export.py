#%%
"""Import libraries.
"""
import numpy as np
from skl2onnx.convert import to_onnx
from onnx import ModelProto
from sklearn_plugins.cluster import SphericalKMeans
import traceback
import onnxruntime as rt

#%%
"""Make synthetic data.
"""
npoints = 200
rA = 5 + np.random.normal(size=npoints, scale=1)
rB = 7 + np.random.normal(size=npoints, scale=2)
rC = 5 + np.random.normal(size=npoints, scale=1)
phiA = np.pi / 4.0 + np.random.normal(size=npoints, scale=np.pi / 8.0)
phiB = -np.pi / 4.0 + np.random.normal(size=npoints, scale=np.pi / 32.0)
phiC = 5 / 4.0 * np.pi + np.random.normal(size=npoints, scale=np.pi / 8.0)
pointsA = np.transpose(np.array([rA * np.cos(phiA), rA * np.sin(phiA)]))
pointsB = np.transpose(np.array([rB * np.cos(phiB), rA * np.sin(phiB)]))
pointsC = np.transpose(np.array([rC * np.cos(phiC), rA * np.sin(phiC)]))
print(pointsA.shape)
print(pointsB.shape)
print(pointsC.shape)
#%%
"""Combine different categories into one dataset.
"""
X: np.ndarray = np.vstack((pointsA, pointsB, pointsC))
print(X.shape)
#%%
"""Perfom Spherical K-Means
"""
skm = SphericalKMeans(n_clusters=3,
                      n_components=0.80,
                      normalize=True,
                      standardize=False)
skm.fit(X)
#%%
"""
Bugs in skl2onnx limits input to be float32 only
"""
onx: ModelProto
try:
    onx = to_onnx(skm, X.astype(np.float32), target_opset=13)
except Exception as e:
    traceback.print_exception(type(e), e, e.__traceback__)
#%%
projection = skm.transform(X)
labels = skm.predict(X)
print(str.format("projection.shape = {}", projection.shape))
print(str.format("labels.shape = {}", labels.shape))
#%%
sess = rt.InferenceSession(onx.SerializeToString())
results = sess.run(None, {'X': X.astype(np.float32)})
#%%
print("expected label")
print(labels)
print("onnx label")
print(results[0])
print(str.format("expected == onnx ? {}", np.array_equal(labels, results[0])))
print("expected projection")
print(projection)
print("onnx projection")
print(results[1])
diff = np.sum(projection.flatten() - results[1].flatten())
print(str.format("diff = {}", diff))
