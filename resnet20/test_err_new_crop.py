from megskull.graph import FpropEnv
from meghair.utils.io import load_network
from megskull.graph import Function
import numpy as np
import cv2

net = load_network(open("./data/resnet20.data_acc91.14", "rb"))
test_func = Function().compile(net.outputs[0])

def load_data(name):
	import pickle
	with open(name, "rb") as fo:
		dic = pickle.load(fo, encoding = "bytes")
	return dic

dic = load_data("/home/liuyanyi02/CIFAR/cifar-10-batches-py/test_batch")
data = dic[b'data']
label = dic[b'labels']

import pickle
with open("meanstd.data", "rb") as f:
	mean, std = pickle.load(f)

raw_data = data.copy()
raw_data = np.resize(raw_data, (10000, 3, 32, 32))

data = (data - mean) / std
data = np.resize(data, (10000, 3, 32, 32))

pred = np.array(test_func(data = data))
err = []
for i in range(len(pred)):
	if pred[i][label[i]] < 0.5:
		err.append(i)

for i in err:
	img = data[i]
	raw_img = raw_data[i]
	inp = []
	inp1 = []
	for a in range(4):
		for b in range(32 - 4, 32):
			for c in range(4):
				for d in range(32 - 4, 32):
					img1 = img.copy()
					mask = np.zeros((3, 32, 32))
					mask[:,a:b,c:d] = 1
					img1 = img1 * mask
					inp.append(img1)
					inp1.append((raw_img * mask).astype(np.uint8))
	pred = test_func(data = inp)
	lis = [(pred[j][label[i]], j) for j in range(len(pred))]
	lis = list(sorted(lis))
	print(label[i])
	for j in range(len(lis) - 5, len(lis))[::-1]:
		k = lis[j]
		print(k[0])
		img = inp1[k[1]]
		img = img.transpose(1, 2, 0)
		img = cv2.resize(img, (512, 512))
		cv2.imshow('x', img)
		cv2.waitKey(0)
