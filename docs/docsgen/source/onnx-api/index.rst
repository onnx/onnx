Modules
=======

.. doctest
	from onnx.defs import onnx_opset_version
	from skl2onnx import to_onnx
	import numpy
	import matplotlib.pyplot as plt
	from sklearn.ensemble import IsolationForest
	from sklearn.datasets import make_blobs

	X, y = make_blobs(n_samples=100, n_features=2)

	model = IsolationForest(n_estimators=3)
	model.fit(X)
	labels = model.predict(X)

	fig, ax = plt.subplots(1, 1)
	for k in (-1, 1):
		ax.plot(X[labels == k, 0], X[labels == k, 1], 'o', label="cl%d" % k)
	ax.set_title("Sample")
	print("This worked!")

.. testoutput
	This worked!

.. autosummary::
   :toctree: modules

   onnx.checker
   onnx.compose
   onnx.external_data_helper
   onnx.helper
   onnx.hub
   onnx.numpy_helper
   onnx.parser
   onnx.utils
   onnx.version_converter
   onnx.version.version