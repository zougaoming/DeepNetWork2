import numpy as np
from .Batch2ConvMatrix import *
import GateC
import datetime
class BackwardList:
	def __init__(self):
		self.Gates = []
		self.Threads = []

	def cleanGateTimes(self):
		self.Gates.clear()

	def run(self):
		for g in self.Gates[::-1]:
			g.backward()

	def printGateTimes(self):
		for g in self.Gates:
			print(g)

	def updateGateTimes(self, g):
		self.Gates.append(g)

class Gate(object):
	def init(self,g=None,Name='',Nework=None,Input='',Input1='',Input2='',key=0,o='',W='',bias='',activeFunc = None):
		#print(Name)
		self.Nework = Nework
		self.param = self.Nework.param
		self.Name = Name
		self.Input = 'self.Nework.' + Input
		self.Input1 = 'self.Nework.' + Input1
		self.Input2 = 'self.Nework.' + Input2
		self.o = o
		self.dz = 'self.Nework.d' + o
		self.dInput = 'd' + Input
		self.dInput1 = 'd' + Input1
		self.dInput2 = 'd' + Input2
		self.W = 'self.param.' + W
		self.bias = 'self.param.' + bias

		self.activeFunc = activeFunc
		self.dw = 'd' + W
		self.dbias = 'd' + bias
		self.upW = W
		self.upBias = bias
		self.key = key
		if g is not None:self.Nework.backwardList.updateGateTimes(g)


class AddGate(Gate):
	def __init__(self,Network,Input1=None,Input2=None,o=None):
		super(self.__class__, self).init(self,'AddGate', Network,Input1=Input1, Input2=Input2, o = o)
	def forward(self):
		setattr(self.Nework, self.o, eval(self.Input1 + '+' + self.Input2))

	def backward(self):
		#setattr(self.Nework,self.dInput1,eval(self.dz))
		#setattr(self.Nework,self.dInput2,eval(self.dz))
		doutput1 = eval(self.dz) * np.ones_like(eval(self.Input1))
		doutput2 = eval(self.dz) * np.ones_like(eval(self.Input2))
		setattr(self.Nework, self.dInput1, doutput1)
		setattr(self.Nework, self.dInput2, doutput2)

class NeuronGate(Gate):
	def __init__(self,Network,Input=None,W=None,bias=None,o=None,activeFunc = None,key=0):
		super(self.__class__, self).init(self,'NeuronGate', Network, Input=Input, o=o,W=W,bias=bias,activeFunc=activeFunc,key=key)
		self.w1 = np.zeros_like(eval(self.W))
		self.w2 = np.zeros_like(self.w1)
		self.w3 = np.zeros_like(self.w1)
		self.b1 = np.zeros_like(eval(self.bias))
		self.b2 = np.zeros_like(self.b1)
		self.b3 = np.zeros_like(self.b1)
		self.adam_t = 0
	def forward(self):
		starttime = datetime.datetime.now()
		self._output = GateC.NeuronGate_Forward(
			np.array(eval(self.Input), dtype=np.float64),
			np.array(eval(self.W), dtype=np.float64),
			np.array(eval(self.bias), dtype=np.float64),
			self.activeFunc.__class__.__name__,
			self.key
		)
		endtime = datetime.datetime.now()

		setattr(self.Nework,self.o,self._output)

	def backward(self):


		self.dw_V, self.dbias_V, dx = GateC.NeuronGate_Backward(
																   np.array(eval(self.dz), dtype=np.float64),
																   self.activeFunc.__class__.__name__,
																	self.param.Optimizer.__class__.__name__,
																	self.key
																   )





		setattr(self.param,self.dbias,self.dbias_V)
		setattr(self.param,self.dw,self.dw_V)
		setattr(self.Nework,self.dInput,dx)
		#print('dw->',dw)
		#self.update(dw,dbias)
		#t = threading.Thread(target=self.update)
		#t.daemon(True)
		#t.start()
		#self.update()
		#return dw,dx
	def update(self):
		self.w1, self.w2, self.w3,self.b1, self.b2,self.b3,self.adam_t, = self.param.Optimizer.Update(self.dw_V,self.upW,self.dbias_V,self.upBias,self.w1,self.w2,self.w3,self.b1,self.b2,self.b3,self.adam_t)

class MulGate(Gate):
	def __init__(self,Network,Input1=None,Input2=None,o=None):
		super(self.__class__, self).init(self,'MulGate', Network, Input1=Input1, Input2=Input2, o=o)
	def forward(self):
		setattr(self.Nework, self.o,np.multiply(eval(self.Input1),eval(self.Input2)))

	def backward(self):
		dInput1 = np.multiply(eval(self.dz),eval(self.Input2))
		dInput2 = np.multiply(eval(self.dz),eval(self.Input1))
		setattr(self.Nework,self.dInput1,dInput1)
		setattr(self.Nework, self.dInput2, dInput2)

class InOutGate(Gate):
	def __init__(self,Network,Input=None,o=None,activeFunc = None):
		super(self.__class__, self).init(self,'InOutGate', Network, Input=Input, o=o,activeFunc=activeFunc)
	def forward(self):
		if self.activeFunc is not None:
			self._output = self.activeFunc.forward(eval(self.Input))
		else:
			self._output = eval(self.Input)
		setattr(self.Nework,self.o,self._output)

	def backward(self):
		if self.activeFunc is not None:
			d = np.multiply(self.activeFunc.backward(self._output),eval(self.dz))
		else:
			d = eval(self.dz)
		setattr(self.Nework,self.dInput,d)

class ConcateGate(Gate):
	def __init__(self, Network,  Input1=None, Input2=None,o=None):
		super(self.__class__, self).init(self,'ConcateGate', Network, Input1=Input1, Input2=Input2,o=o)
	def forward(self):
		setattr(self.Nework,self.o,np.concatenate((eval(self.Input1),eval(self.Input2)),axis=0))

	def backward(self):
		l = eval(self.Input1).shape[0]
		setattr(self.Nework, self.dInput1, eval(self.dz)[:l])
		setattr(self.Nework, self.dInput2,eval(self.dz)[l:])

class CopyGate(Gate):
	def __init__(self, Network, Input=None, o=None):
		super(self.__class__, self).init(self,'CopyGate', Network, Input=Input, o=o)

	def forward(self):
		setattr(self.Nework, self.o, eval(self.Input))
		if hasattr(self.Nework, self.dInput) is True:  #
			setattr(self.Nework, self.dInput, np.zeros_like(eval(self.dz)))

	def backward(self):

		if hasattr(self.Nework, self.dInput) is False:
			setattr(self.Nework, self.dInput, eval(self.dz))
		else:
			setattr(self.Nework, self.dInput, eval(self.dz) + eval('self.Nework.' + self.dInput))

class FlattenGate(Gate):
	def __init__(self, Network, Input=None, o=None):
		super(self.__class__, self).init(self,'FlattenGate', Network, Input=Input, o=o)

		self.H = 0
		self.W = 0
		self.input_dim = 0
	def forward(self):
		self.input = eval(self.Input)
		self.N = self.input.shape[0]
		l = 1
		for s in self.input.shape[1:]:
			l = l * s
		out = np.zeros((l,self.N))
		for n in range(self.N):
			out[:,n] = self.input[n].flatten()
		self._output = out
		setattr(self.Nework, self.o, self._output)

	def backward(self):
		dz = eval(self.dz)
		dinput = dz.reshape(-1,self.N)
		out = np.zeros_like(self.input)
		for n in range(self.N):
			out[n] = dinput[:,n].reshape(self.input.shape[1],self.input.shape[2],self.input.shape[3])

		setattr(self.Nework, self.dInput, out)

class CNNGate(Gate):
	def __init__(self, Network, Input=None,W=None,bias=None,o=None,activeFunc = None,fliters=3,step = 1,padding = 0,channel_in = 1,channel_out = 1,key=0):
		super(self.__class__, self).init(self,'CNNGate', Network, Input=Input, o=o, W=W, bias=bias,
											 activeFunc=activeFunc,key=key)
		self.fliters = fliters
		self.step = step
		self.padding = padding
		self.channel_in = channel_in
		self.channel_out = channel_out

		self.w1 = np.zeros_like(eval(self.W))
		self.w2 = np.zeros_like(self.w1)
		self.w3 = np.zeros_like(self.w1)
		self.b1 = np.zeros_like(eval(self.bias))
		self.b2 = np.zeros_like(self.b1)
		self.b3 = np.zeros_like(self.b1)
		self.adam_t = 0

	def getOutputSize(self, M, S, F, P):
		return (M - F + 2 * P) / S + 1



	def conv(self, input, weight, outputH, outputW, step):
		result = np.zeros((outputH, outputW))
		fliter = weight.shape[0]
		for h in range(outputH):
			for w in range(outputW):
				i_a = input[h * step:h * step + fliter, w * step:w * step + fliter]
				result[h][w] = np.multiply(i_a,weight).sum()  # self.calc_connv(i_a,weight)#np.multiply(i_a,weight).sum()#self.calc_connv(i_a,weight)
				#print("python h=", h, "w=", w, "sum=", result[h][w])
		return result

	def paddingZeros(self, input, P):
		if P > 0:
			if input.ndim == 2:
				input = np.pad(input, ((P, P), (P, P)), 'constant')
			elif input.ndim == 4:
				input = np.pad(input, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')
		return input
	def forward(self):
		starttime = datetime.datetime.now()
		print(self.activeFunc.__class__.__name__)
		self._output = GateC.CnnGate(np.array(eval(self.Input),dtype=np.float64),
									 np.array(eval(self.W),dtype=np.float64),
									 np.array(eval(self.bias),dtype=np.float64),
									 self.activeFunc.__class__.__name__,
									 self.step,
									 self.padding,self.key
									 )
		endtime = datetime.datetime.now()
		#print('cnn zjCforwardtime->', endtime-starttime)

		setattr(self.Nework, self.o, self._output)
		return self._output

	def backward(self):
		starttime = datetime.datetime.now()
		self.dw_V,self.dbias_V,dx = GateC.CnnGate_Backward(
														   np.array(eval(self.dz), dtype=np.float64),
														   self.key,self.activeFunc.__class__.__name__,self.param.Optimizer.__class__.__name__,)
		endtime = datetime.datetime.now()
		#print('cnn backwardtime->',endtime - starttime)
		setattr(self.param, self.dbias, self.dbias_V)
		setattr(self.param, self.dw, self.dw_V)
		setattr(self.Nework, self.dInput, dx)
		#self.update()
	def update(self):

		self.w1, self.w2, self.w3, self.b1, self.b2, self.b3, self.adam_t, = self.param.Optimizer.Update(self.dw_V,
																												self.upW,
																										 self.dbias_V,
																												self.upBias,
																												self.w1,
																												self.w2,
																												self.w3,
																												self.b1,
																												self.b2,
																												self.b3,
																												self.adam_t)

class PoolGate(Gate):
	def __init__(self, Network, key=0,Input=None,W=None,bias=None,o=None,activeFunc = None,fliters=2,step = 1,padding=0,F_num = 1,type='MAX'):
		super(self.__class__, self).init(self,'PoolGate', Network, Input=Input, o=o,key=key,
											 activeFunc=activeFunc)
		self.fliters = fliters
		self.step = step
		self.type = type


		#GateC.PoolGate_init(np.array(eval(self.Input), dtype=np.float64), self.fliters, self.step,self.type, self.key)

	def getOutputSize(self, M, S, F):
		return (M - F) / S + 1

	def calc_pool3(self, input, type, index, f=0, c=0):
		result = .0
		if type == 'MAX':
			result = np.max(input)
			idx = np.argmax(input)
			self.bz_x[f][c][index] = int(idx / input.shape[0])
			self.bz_y[f][c][index] = idx - (self.bz_x[f][c][index] * input.shape[0])
		elif type == 'AVERAGE':
			result = np.average(input)
		return result

	def calc_pool(self, input, type, index, f=0, c=0):
		result = .0
		if type == 'MAX':
			max = input[0, 0]
			for i in range(input.shape[0]):
				for j in range(input.shape[1]):
					if input[i, j] > max:
						max = input[i, j]
						self.bz_x2[f][c][index] = i
						self.bz_y2[f][c][index] = j

			result = max
		elif type == 'AVERAGE':
			size = input.shape[0] + input.shape[1]
			for i in range(input.shape[0]):
				for j in range(input.shape[1]):
					result += input[i, j] / size
		return result
	def forward(self):
		'''
		input = eval(self.Input)
		inputW = input.shape[3]
		inputH = input.shape[2]
		self.outputW = int(self.getOutputSize(inputW, self.step, self.fliters))
		self.outputH = int(self.getOutputSize(inputH, self.step, self.fliters))
		self.FNum = input.shape[0]
		self.channel = input.shape[1]
		inputW = input.shape[3]
		inputH = input.shape[2]
		self.outputW = int(self.getOutputSize(inputW, self.step, self.fliters))
		self.outputH = int(self.getOutputSize(inputH, self.step, self.fliters))
		self._output = np.zeros((self.FNum, self.channel, self.outputH, self.outputW))
		self.bz_x2 = np.zeros((self.FNum, self.channel, self.outputW * self.outputH))
		self.bz_y2 = np.zeros((self.FNum, self.channel, self.outputW * self.outputH))
		for f in range(self.FNum):
			for c in range(self.channel):
				index = 0
				for h in range(self.outputH):
					for w in range(self.outputW):
						i_a = input[f][c][h * self.step:h * self.step + self.fliters,
							  w * self.step:w * self.step + self.fliters]
						self._output[f][c][h][w] = self.calc_pool(i_a, self.type, index, f, c)
						index += 1
		'''
		input = eval(self.Input)
		starttime = datetime.datetime.now()
		if self.type == 'MAX':
			self._output,self.bz_x,self.bz_y = GateC.PoolGate(np.array(input,dtype=np.float64), self.fliters, self.step,self.type,self.key)
		else:
			self._output = GateC.PoolGate(input, self.fliters, self.step, self.type)
		endtime = datetime.datetime.now()
		print('Pool forwardtime->', endtime - starttime)
		#print(self.bz_x.shape,self.bz_x2.shape)
		#print(self.bz_y.shape,self.bz_y2.shape)
		#print('duibi0->',self.bz_x - self.bz_x2)
		#print('duibi1->', self.bz_y - self.bz_y2)
		setattr(self.Nework, self.o, self._output)
	def backward(self):

		starttime = datetime.datetime.now()
		result = GateC.PoolGate_Backward(np.array(eval(self.dz),dtype=np.float64),
										self.key)
		endtime = datetime.datetime.now()
		#print('Pool backwardtime->', endtime - starttime)
		setattr(self.Nework, self.dInput, result)


		'''
		input = eval(self.Input)
		inputW = input.shape[3]
		inputH = input.shape[2]
		self.outputW = int(self.getOutputSize(inputW, self.step, self.fliters))
		self.outputH = int(self.getOutputSize(inputH, self.step, self.fliters))
		self.FNum = input.shape[0]
		self.channel = input.shape[1]
		dz = eval(self.dz)
		result = np.zeros((input.shape))
		for f in range(input.shape[0]):
			for c in range(input.shape[1]):
				if self.type == 'MAX':
					index = 0
					for i in range(self.outputH):
						for j in range(self.outputW):
							x = int(i * self.step + self.bz_x[f][c][index])
							y = int(j * self.step + self.bz_y[f][c][index])
							tmp = dz[f][c][i][j]
							#print(result.shape,f,c,x,y, self.bz_y[f][c][index],i,j)
							result[f][c][x][y] = tmp
							index += 1

				elif self.type == 'AVERAGE':
					for i in range(self.outputW):
						for j in range(self.outputH):
							pool_size = self.step * self.step
							for m in range(self.step):
								for n in range(self.step):
									result[f][c][i * self.step + m][j * self.step + n] = dz[f][c][i][j] / pool_size
		setattr(self.Nework, self.dInput, result)
		#print('result->',result2 - result)
		'''