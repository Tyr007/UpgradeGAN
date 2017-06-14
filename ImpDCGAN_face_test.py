'''
test the net on database. almost the same as train.

Author:Tyr

Project:

Denpendencies: tensorflow1.0, lassagne and keras 2.0 

Usage: python2 

'''

import numpy as np

import time
#import form_input_test as form input
import form_input



from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten, Reshape

from keras.layers import Conv2D, Conv2DTranspose

from keras.layers import UpSampling2D, LeakyReLU, Dropout

from keras.layers import BatchNormalization

from keras.optimizers import Adam, RMSprop



import matplotlib.pyplot as plt





class ImpDCGAN(object):

	def __init__(self, img_rows=64, img_cols=64, channel=1):

		self.img_rows = img_rows

		self.img_cols = img_cols

		self.channel = channel

		self.D = None

		self.G = None

		self.AM= None

		self.DM= None



	def discriminator(self): 

		

		if self.D:

			return self.D

		self.D= Sequential()

		

		dropout=0.4

		

		input_shape=(self.img_rows, self.img_cols, self.channel)
		self.D.add(Conv2D(32,5,strides=2,padding='same',input_shape=input_shape,activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#32 32
		self.D.add(Conv2D(32,5,strides=2,padding='same',activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#16 16
	
		self.D.add(Conv2D(64,5,strides=2,padding='same',activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#8 8
		self.D.add(Conv2D(64,3,strides=2,padding='same',activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#4 4
		self.D.add(Flatten())
		self.D.add(Dense(512))
		#out to one-hot vector + fake labels

		self.D.add(Dense(7))

		self.D.add(Activation('sigmoid'))

		self.D.summary()

		return self.D







	def generator(self):

		if self.G:

        		return self.G

		self.G = Sequential()

		dropout =0.4


		self.G.add(Dense(4*4*128, input_dim=106))

		self.G.add(BatchNormalization(momentum=0.9))

		
		self.G.add(Dropout(dropout))
		self.G.add(Reshape((4,4,128)))
		#end of dim change
		self.G.add(Conv2DTranspose(64,2,strides=2,padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))

		#self.G.add(UpSampling2D())#8 8
		self.G.add(Conv2DTranspose(64,2,strides=1,padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))

		
		self.G.add(Conv2DTranspose(32,2,strides=2,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9)) #16 16
	
		self.G.add(Conv2DTranspose(32,2,strides=1,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9)) #16 16

		self.G.add(Conv2DTranspose(32,2,strides=2,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Conv2DTranspose(32,2,strides=1,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9)) #16 16


		self.G.add(Conv2DTranspose(32,2,strides=2,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9))
	
		self.G.add(Conv2DTranspose(1,1,padding='same'))

		self.G.add(Activation('sigmoid'))

		self.G.summary()

		return self.G



	def discriminator_model(self):

		if self.DM:

			return self.DM

		optimizer=RMSprop(lr=0.0008,clipvalue=1.0,decay=0.00000002)

		self.DM=Sequential()

		self.DM.add(self.discriminator())

		self.DM.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

		return self.DM



	def adversarial_model(self):

		if self.AM:

			return self.AM

		optimizer=RMSprop(lr=0.0004,clipvalue=1.0,decay=0.00000002)

		self.AM=Sequential()

		self.AM.add(self.generator())

		self.AM.add(self.discriminator())

		self.AM.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

		return self.AM



class Improved_DCGAN(object):

	def __init__(self):

		self.img_rows =64

		self.img_cols = 64

		self.channel = 1

		self.x_train = form_input.input_x()

			#data type: [number,x,y,channel] of 0.0-1.0 

		self.x_sum = form_input.input_sum()

		self.y_train = form_input.input_y()

		self.typesum=6
		self.fakesum=1

			#fake map is 1 and true is 0. so it's half GAN half SVM

		#hope Form_input give the right types.

		

		self.DCGAN=ImpDCGAN()

		self.discriminator = self.DCGAN.discriminator_model()

		self.adversarial = self.DCGAN.adversarial_model()

		self.generator = self.DCGAN.generator()


	def save(self):

		self.generator.save_weights('Gweight.h5')

		self.adversarial.save_weights('Aweight.h5')

		self.discriminator.save_weights('Dweight.h5')
	def load(self):
                self.generator.load_weights('Gweight.h5')
                self.adversarial.load_weights('Aweight.h5')
                self.discriminator.load_weights('Dweight.h5')

	
	def show_ans(self, save2file=True, fake=True, samples=16,label=0, step=0):

		if fake:
			noise = np.random.uniform(-1.0,1.0,size=[samples,100])
			Ulabel=np.zeros([samples,self.typesum]) #Ulabel: 7 types one-Hot
			for k in range(samples):
				Ulabel[k,label]=1

			Unoise=np.zeros([samples,106])
			Unoise[:,:100]=noise
			Unoise[:,100:]=Ulabel
			filename='ans/ImpG_%d.png' %label

			images=self.generator.predict(Unoise)

			lig=self.discriminator.predict(images)

			

		else:

			i=np.random.randint(0,self.x_sum,samples)

			images=self.x_train[i,:, :, :]

			filename='ans/ImpData_%d.png' %label

			lig=self.discriminator.predict(images)

			

		plt.figure(figsize=(10,10))

		for i in range(samples):

			plt.subplot(4,4,i+1)

			image=images[i,:,:,:]

			image=np.reshape(image,[self.img_rows,self.img_cols])

			plt.imshow(image,cmap='gray')

			plt.axis('off')

		plt.tight_layout()

		if save2file:

			plt.savefig(filename)

			plt.close('all')

			if fake :

				typefile='ans/ImgFL_%d.txt' %label

			else :

				typefile='ans/ImgDL_%d.txt' %label

			fp2=open(typefile,'w')

			fp2.write(str(lig))

			fp2.close()

		else:

			plt.show()

			print lig

	



if __name__ == '__main__':

	savestep=700

	mainstep=100000

	printat=100

	fp=open('savedata.txt','a')

	fp.close()

	fp=open('savedata.txt','r+')

	st=fp.read()

	if st=='' :

		fp.write('0')

		st='0'

	fp.close()

	i=int(st)

	if i==0 :

		print 'No saved data, new module start.'

	else :

		print 'Saved data load. Now step:', i

	IDC=Improved_DCGAN()

	IDC.load()

	for k in range(6):
		IDC.show_ans(save2file=True, fake=True, samples=16,label=k, step=0)
	
