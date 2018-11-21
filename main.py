from utils import max_permutations
from generators import JigsawGenerator
from models import AlexNet, CFN
from datetime import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import optimizers

import numpy as np
np.seterr(all="raise")

batch_size = 128

P = max_permutations()

TRAIN_PATH = "/media/kashgar/data/pnn_training/p256/train/se/"
VALID_PATH = "/media/kashgar/data/pnn_training/p256/validation/se/"

train_gen = JigsawGenerator(TRAIN_PATH, batch_size, 2, P)
valid_gen = JigsawGenerator(VALID_PATH, batch_size, 0, P)

alexnet = AlexNet()
model = CFN(alexnet)
#opt = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
opt = optimizers.Adam(lr=0.00001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['categorical_accuracy'])

now = datetime.now().strftime('%Y%m%d%H%M%S')

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint('logs/{now}/'.format(now=now) + 'epoch_{epoch:04d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='logs/{now}'.format(now=now), batch_size=batch_size, write_grads=True, write_images=True)
reduce_lr = ReduceLROnPlateau(verbose=1, patience=3)

model.fit_generator(
    train_gen,
    epochs = 1000,
    validation_data = valid_gen,
    callbacks=[early_stopping, checkpoint, tb, reduce_lr],
    use_multiprocessing=True,
    shuffle=True,
    workers=10,
    max_queue_size=50)

