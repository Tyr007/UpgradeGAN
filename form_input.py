'''
form_input
formatting input for Improved GAN

Functions:
	input_x(), give train data like [0:1000,0:32,0:24,0:1] of real in 0.0-1.0
	input_sum(), sum of the full train dataset
	input_y(), the label of dataset, and 1 for no label as realdata
		like [0:1000,0:5] as one-hot vectors.

'''
import numpy as npy

def input_x():
	t=npy.load('fdata.npy')
	zx=npy.ones([len(t),240,180,1])
		
	zx=t
	return zx

def input_sum():
	z=npy.load('flabel.npy')
	return len(z)

def input_y():
	y=npy.load('flabel.npy')
	zy=npy.zeros([len(y)])
	for i in range(len(y)):
		zy[i]=y[i]
	return zy

if __name__ == '__main__':
	print 'Only use for formatting input to Improved_DCGAN'
	print 'Do not open and run it.'
	exit()
