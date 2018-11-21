from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from keras.models import Model


def AlexNet(tilesize=64):
	inputTensor = Input((tilesize, tilesize, 3))
	x = Conv2D(96, (11, 11), strides=(2, 2), padding='same')(inputTensor)
	x = Activation("selu")(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	# lrn

	x = Conv2D(256, (5, 5), padding='valid')(x)
	x = Activation("selu")(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	# lrn

	x = Conv2D(384, (3, 3), padding='same')(x)
	x = Activation("selu")(x)

	x = Conv2D(384, (3, 3), padding='same')(x)
	x = Activation("selu")(x)

	x = Conv2D(256, (3, 3), padding='same')(x)
	x = Activation("selu")(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

	x = Flatten()(x)
	x = Dense(1024)(x)
	x = Activation("selu")(x)

	model = Model(inputTensor, x, name='AlexNet')
	return model

'''
def AlexNet(tilesize=64):
	inputTensor = Input((tilesize, tilesize, 3))
	x = Conv2D(96, (11, 11), strides=(2, 2), padding='same')(inputTensor)
	x = Activation("selu")(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	# lrn

	x = Conv2D(256, (5, 5), padding='valid')(x)
	x = Activation("selu")(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	# lrn

	x = Conv2D(384, (3, 3), padding='same')(x)
	x = Activation("selu")(x)

	x = Conv2D(384, (3, 3), padding='same')(x)
	x = Activation("selu")(x)

	x = Conv2D(256, (3, 3), padding='same')(x)
	x = Activation("selu")(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(128, (1, 1), padding='same')(x)
	x = Activation("selu")(x)
	x = Conv2D(64, (1, 1), padding='same')(x)
	x = Activation("selu")(x)

	x = Flatten()(x)
	x = Dense(1024)(x)
	x = Activation("selu")(x)

	model = Model(inputTensor, x, name='AlexNet')
	return model
'''

def CFN(model, tilesize=64, permutations=100):
	inputs = [Input((tilesize, tilesize, 3)) for x in range(9)]
	tower_outputs = [model(input_) for input_ in inputs]
	x = Concatenate()(tower_outputs)
	x = Dropout(0.5)(x)
	x = Dense(4096)(x)
	x = Activation("selu")(x)
	x = Dropout(0.5)(x)
	x = Dense(permutations)(x)
	predictions =  Activation("softmax")(x)
	cfn = Model(inputs, predictions, name="CFN")
	return cfn
