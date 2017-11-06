from megskull.graph import FpropEnv
from meghair.utils.io import load_network
from megskull.graph import Function
import numpy as np
import cv2

net = load_network(open("./data/resnet20.data_acc90.4", "rb"))
test_func = Function().compile(net.outputs[0])

def load_data(name):
	import pickle
	with open(name, "rb") as fo:
		dic = pickle.load(fo, encoding = "bytes")
	return dic

dic = load_data("/home/liuyanyi02/CIFAR/cifar-10-batches-py/test_batch")
data = dic[b'data']
label = dic[b'labels']

data = np.resize(data, (10000, 3, 32, 32))

idx = 0
img = data[idx]
inp = []
for i in range(32):
	for j in range(i + 13, 32):
		for k in range(32):
			for l in range(k + 13, 32):
				img1 = img.copy()
				mask = np.zeros((3, 32, 32))
				mask[:,i:j,k:l] = 1
				img1 = img1 * mask
				noise = (128**0.5) * np.random.randn(3, 32, 32) + 128
				noise = noise * (1 - mask)
				img1 = img1 + noise
				img1 = img1.astype(np.uint8)
				inp.append(img1)
"""
import cv2
for i in range(10):
	img = data[i].transpose(1, 2, 0)
	img = img[:,::-1,:]
	cv2.imshow('x', img)
	cv2.waitKey(0)
"""
inp1 = inp.copy()
inp = np.array(inp)
inp = inp.astype(np.float32)
inp = (inp - 128) / 256

pred = test_func(data = inp)
#print(np.array(pred).shape)
#pred = np.argmax(np.array(pred), axis = 1)
lis = []
for i in range(len(inp1)):
	lis.append((pred[i][label[idx]], i))
lis = sorted(lis)
print(label[idx])
for j in range(len(lis))[::-1]:
	i = lis[j]
	print(i[0])
	inp1[i[1]] = inp1[i[1]].transpose(1, 2, 0)
	cv2.imshow('x', inp1[i[1]])
	cv2.waitKey(0)


