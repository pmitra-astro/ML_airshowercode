import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib.backends.backend_agg import FigureCanvasAgg
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from matplotlib import cm
import cPickle
from keras.datasets import cifar10
import imagetoarray as process
##########################################################################################






##########################################################################################
datafile=cPickle.load(open('input_footprint_test.dat','r'))
xmax=datafile['xmax']
x_data=[]
y_data=[]
x_pos=datafile['x_pos']
y_pos=datafile['y_pos']
power=datafile['power']


###sort acc to xmax
for i in np.arange(150):
	img=plt.imread('images_zen0/img_{0}.png'.format(i))
	img= img_to_array(img)#process.imagetoarray(x_pos[i],y_pos[i],power[i])
	x_data.append(img)
	x_dataall=np.asarray(x_data)
        

x_train=x_dataall[:140]
y_train=xmax[:140]
x_test=x_dataall[140:]
y_test=xmax[140:]


model = Sequential([
    Conv2D( 32, (3, 3), activation='relu', input_shape=(480,640,4)),
    #Dropout(drop),
    Conv2D(64, (3, 3),activation='relu'),
    #Dropout(drop),
    Conv2D(64, (3,3),activation='relu'),
    #Dropout(drop),
    Flatten(),
    Dense(64, activation='relu'),
    #Dropout(drop),
    Dense(1, activation='softmax')
])
	


#################################################################################
#model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1E-3))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

results=model.fit(x_train,y_train,batch_size=32,epochs=4,validation_split=0.05)
'''
model.evaluate(x_test,y_test)
y_predicted=model.predict(x_test)
y_predicted=y_predicted.reshape(-1)


print y_predicted

#print "differences", y_test-y_predicted



epoch_nos=np.arange(1,11)
plt.plot(epoch_nos,results['loss'],'bo',legend='loss')
plt.plot(epoch_nos,results['val_loss'],'ro',legend='validation loss')
plt.legend()
plt.show()
'''
