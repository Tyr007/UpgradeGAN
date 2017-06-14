'''

Improved DCGAN, to generate and classify face images with different emotions.

Author:Tyr


Denpendencies: tensorflow1.0, lassagne and keras 2.0 

Usage: python2 

'''

import numpy as np

import time

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
		self.D.add(Conv2D(32,5,strides=1,padding='same',input_shape=input_shape,activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#64 64 32


		self.D.add(Conv2D(32,5,strides=2,padding='same',activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#32 32 32

		self.D.add(Conv2D(32,5,strides=1,padding='same',activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#32 32 32
		self.D.add(Conv2D(32,5,strides=2,padding='same',activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#16 16 32
	
		self.D.add(Conv2D(64,5,strides=1,padding='same',activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#16 16 64
		self.D.add(Conv2D(64,3,strides=2,padding='same',activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#8 8 64
		self.D.add(Conv2D(128,3,strides=2,padding='same',activation=LeakyReLU(alpha=0.02)))
		self.D.add(Dropout(dropout))#4 4 128
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

		#self.G.add(UpSampling2D())#8 8 64
		self.G.add(Conv2DTranspose(64,2,strides=1,padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))

		
		self.G.add(Conv2DTranspose(32,2,strides=2,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9)) #16 16 32
	
		self.G.add(Conv2DTranspose(32,2,strides=1,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9)) 

		self.G.add(Conv2DTranspose(32,2,strides=2,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9))# 32 32 32
		self.G.add(Conv2DTranspose(32,2,strides=1,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9)) 


		self.G.add(Conv2DTranspose(32,2,strides=2,padding='same'))

		self.G.add(BatchNormalization(momentum=0.9))#64 64 32
	
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

			#Real image have fakelabel as 0, and Fake image's label is 1

		

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

	def train(self, step=0, saved=False, train_steps=100, batch_size=128, save_interval=0):

		if saved == True :

			self.load()

		i=0

		# everytime use train will train it by train_steps. so (mainstep mod Train_steps==0) is better (not must)

		while i<train_steps :

			data_list=np.random.randint(0,self.x_sum,batch_size)

			#the list of this batch;

			images_train = self.x_train[data_list, :, :, :]

			noise = np.random.uniform(-1.0,1.0,size=[batch_size,100])
			Ulabel=np.zeros([batch_size,self.typesum]) #Ulabel: Upsize from label to one-hot
			Llabel=np.zeros([batch_size,self.typesum]) #Little label with changed weight
			for k in range(batch_size):
				Ulabel[k,int((noise[k,1]+1)/2*self.typesum)]=1
				Llabel[k,int((noise[k,1]+1)/2*self.typesum)]=min(0.1+0.9*step/20000,0.6)
                        #upper label means classify more clearly.
			#Ulabel: [batch_size, 6] Unoise [batch_size,106]
			Unoise=np.zeros([batch_size,106])
			Unoise[:,:100]=noise
			Unoise[:,100:]=Ulabel
			
			images_fake=self.generator.predict(Unoise)

			x = np.concatenate((images_train, images_fake))

			y = np.ones([2*batch_size,self.typesum+self.fakesum])
			y[batch_size:, self.typesum:]=0.9
			y[batch_size: , :self.typesum]=Llabel
                        #the first several types is a one-hot vector, the next fake labels are 1 means fake, to train Dnet.
			y[:batch_size,:self.typesum]=0
			for k in range(batch_size):
				if int(self.y_train[data_list[k]]+0.01)<7:
					y[k,int(self.y_train[data_list[k]]+0.01)]=min(0.1+0.9*step/20000,0.7)
				
			y[:batch_size,self.typesum:]=0

			d_loss = self.discriminator.train_on_batch(x,y)
		

			#^train D net

			noise = np.random.uniform(-1.0,1.0,size=[batch_size,100])
			Ulabel=np.zeros([batch_size,self.typesum]) #
			Llabel=np.zeros([batch_size,self.typesum]) #
			for k in range(batch_size):
				Ulabel[k,int((noise[k,1]+1)/2*self.typesum)]=1
				Llabel[k,int((noise[k,1]+1)/2*self.typesum)]=min(0.1+0.9*step/30000,0.3)
			Unoise=np.zeros([batch_size,106])
			Unoise[:,:100]=noise
			Unoise[:,100:]=Ulabel
		

			y = np.zeros([batch_size,self.typesum+self.fakesum])
			y[: , self.typesum:]=0
                        y[: , :self.typesum]=Llabel
                        #the first several types is a one-hot vector, and next fake labels are 0 to train Gnet.

			a_loss=self.adversarial.train_on_batch(Unoise,y)
                        #^train A(G) net.
			log_mesg = "%d : [D loss:%f, acc:%f]  [A loss:%f, acc:%f]" %(i+1+step, d_loss[0], d_loss[1], a_loss[0], a_loss[1])

			print (log_mesg)

			flog=open('logfile.txt','a')

			flog.write(log_mesg)
			flog.write('\n')



			if save_interval>0:

				if (i+1)%save_interval==0:

					self.show_image(save2file=True, samples=16, step=i+1+step)

					self.show_image(save2file=True, fake=False, step=i+1+step)

			

			

			i=i+1

			

	def show_image(self, save2file=False, fake=True, samples=16, step=0):

		if fake:
			noise = np.random.uniform(-1.0,1.0,size=[samples,100])
			Ulabel=np.zeros([samples,self.typesum]) #Ulabel: 7 types one-Hot
			for k in range(samples):
				Ulabel[k,int((noise[k,1]+1)/2*self.typesum)]=1

			Unoise=np.zeros([samples,106])
			Unoise[:,:100]=noise
			Unoise[:,100:]=Ulabel
			filename='save/ImpG_%d.png' %step

			images=self.generator.predict(Unoise)

			lig=self.discriminator.predict(images)

			

		else:

			i=np.random.randint(0,self.x_sum,samples)

			images=self.x_train[i,:, :, :]

			filename='save/ImpData_%d.png' %step

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

				typefile='save/ImgFL_%d.txt' %step

			else :

				typefile='save/ImgDL_%d.txt' %step

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



	while i<mainstep :

		if i==0 :

			IDC.train(saved=False,step=0,train_steps=savestep,batch_size=16,save_interval=printat)

			i=i+savestep

			IDC.save()

			print 'Module saved to file, now step:', i

			fp=open('savedata.txt','w')

			fp.write(str(i))

			fp.close()

		else:

			IDC.train(saved=True,step=i,train_steps=savestep,batch_size=16,save_interval=printat)

			i=i+savestep

			IDC.save()

			print 'Module saved to file, now step:', i

			fp=open('savedata.txt','w')

			fp.write(str(i))

			fp.close()

	print 'train finished'

	
