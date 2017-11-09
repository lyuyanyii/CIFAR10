import pickle
import numpy as np

from ntools.visualizer.curve import Curve, WebCurveVisualizer

with open("../plain30/pca_ener.data", "rb") as f:
	data1 = pickle.load(f)

with open("../plain30_dropout/pca_ener.data", "rb") as f:
	data2 = pickle.load(f)

with open("../plain30_orth/pca_ener.data", "rb") as f:
	data3 = pickle.load(f)


curve = Curve(title="PCA energy", x_label="channel", y_label="yy")
idx = 0

for i, j, k in zip(data1, data2, data3):
	idx += 1
	x = range(len(i))
	if idx > 20 and idx <= 30:
		i /= i[-1]
		j /= j[-1]
		k /= k[-1]
		curve.add(i, x, legend='NoReg{}'.format(idx), group_id=idx)
		curve.add(j, x, legend='Dropout{}'.format(idx), style = '.', group_id = idx)
		curve.add(k, x, legend='Orth{}'.format(idx), style = '--', group_id = idx)

webcurve = WebCurveVisualizer(curve)
context = webcurve.plot()
WebCurveVisualizer.show(context)
