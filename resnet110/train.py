import argparse
from meghair.train.env import TrainingEnv, Action
from megskull.opr.loss import WeightDecay

from megskull.graph import FpropEnv
import megskull
from dpflow import InputPipe, control
import time

from network import make_network
import numpy as np

import msgpack
import msgpack_numpy as m

minibatch_size = 128
patch_size = 32

def get_minibatch(p, size):
	data = []
	labels = []
	for i in range(size):
		#a = p.get()
		#(img, label) = msgpack.unpackb(a, object_hook = m.decode)
		(img, label) = p.get()
		data.append(img)
		labels.append(label)
	return {"data": np.array(data).astype(np.float32), "label":np.array(labels)}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	with TrainingEnv(name = "lyy.resnet20.test", part_count = 2, custom_parser = parser) as env:
		net = make_network(minibatch_size = minibatch_size)
		preloss = net.loss_var
		net.loss_var = WeightDecay(net.loss_var, {"*conv*": 1e-4, "*fc*": 1e-4})
	
		train_func = env.make_func_from_loss_var(net.loss_var, "train", train_state = True)
		valid_func = env.make_func_from_loss_var(net.loss_var, "val", train_state = False)
	
		warmup = False
		if warmup:
			lr = 0.01
		else:
			lr = 0.1
		optimizer = megskull.optimizer.Momentum(lr, 0.9)
		#optimizer.learning_rate = 0.01
		optimizer(train_func)
		
		train_func.comp_graph.share_device_memory_with(valid_func.comp_graph)
	
		dic = {
			"loss": net.loss_var,
			"pre_loss": preloss,
			"outputs": net.outputs[0]
		}
		train_func.compile(dic)
		valid_func.compile(dic)
		
		env.register_checkpoint_component("network", net)
		env.register_checkpoint_component("opt_state", train_func.optimizer_state)
	
		tr_p = InputPipe("lyy.CIFAR10.resnet20.train_mypipe", buffer_size = 1000)
		va_p = InputPipe("lyy.CIFAR10.resnet20.valid", buffer_size = 1000)
		epoch = 0
		EPOCH_NUM = 45000 // 128
		i = 0
		max_acc = 0
		extra_it = 0
	
		import time
		his = []
		with control(io = [tr_p]):
			with control(io = [va_p]):
		
				a = time.time()
				TOT_IT = 80000
				while i <= TOT_IT:
					i += 1
					data = get_minibatch(tr_p, minibatch_size)
					out = train_func(data = data['data'], label = data["label"])
					loss = out["pre_loss"]
					pred = np.array(out["outputs"]).argmax(axis = 1)
					acc = (pred == np.array(data["label"])).mean()
					if i > 10000 and loss > 1.5:
						print("oh my shoulder")
						env.save_checkpoint("resnet110.data.boom")
					his.append([loss, acc])
					print("minibatch = {}, loss = {}, acc = {}, lr = {}".format(i, loss, acc, round(float(optimizer.learning_rate), 5)))
					#Learning Rate Adjusting
					if i == 400 and warmup:
						optimizer.learning_rate = 0.1
					if i == 32000 + extra_it or i == 48000 + extra_it:
						optimizer.learning_rate /= 10
					if i == 64000:
						optimizer.learning_rate = 1e-5
						env.save_checkpoint("resnet110.data.64000")
					if i % (EPOCH_NUM) == 0:
						epoch += 1
						data_val = get_minibatch(va_p, 500)
						out_val = valid_func(data = data["data"], label = data["label"])
						pred = np.argmax(np.array(out["outputs"]), axis = 1)
						acc = (np.array(pred) == np.array(data["label"])).mean()
	
						print("epoch = {}, acc = {}, max_acc = {}".format(epoch, acc, max_acc))
						b = time.time()
						b = b + (b - a) / i * TOT_IT
						print("Expected finish time {}".format(time.asctime(time.localtime(b))))
	
						#if acc > 0.2 and i < 32000:
						#	optimizer.learning_rate = 0.05
						if acc > max_acc and i > 64000:
							max_acc = acc
							env.save_checkpoint("resnet110.data.bestmodel")
						env.save_checkpoint("resnet110.data")
						print("**************************")
						import pickle
						with open("hisloss.data", "wb") as f:
							pickle.dump(his, f)
			
