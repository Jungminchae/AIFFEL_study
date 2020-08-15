import tensorflow as tf
from tensorflow import keras
import pandas as pd 
import numpy as np 
import os

from dataloader import Dataloader,Augmented_dataset
from utils import * 
import argparse

def to_bool(x):
    if x.lower() in ['true','t']:
        return True
    elif x.lower() in ['false','f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Bool 값을 넣으세요')
def create_cnn_model(x_train):
    inputs = tf.keras.layers.Input(x_train.shape[1:])

    bn = tf.keras.layers.BatchNormalization()(inputs)
    conv = tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu')(bn)
    bn = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

    bn = tf.keras.layers.BatchNormalization()(pool)
    conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    bn = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

    flatten = tf.keras.layers.Flatten()(pool)

    bn = tf.keras.layers.BatchNormalization()(flatten)
    dense = tf.keras.layers.Dense(1000, activation='relu')(bn)

    bn = tf.keras.layers.BatchNormalization()(dense)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(bn)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment_data', type=to_bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_save', type=bool, default=True)
    parser.add_argument('--submission_name', type=str, default='submission.csv')
    args = parser.parse_args()
    dl = Dataloader()

    if args.augment_data ==True:
        aug = Augmented_dataset()
    
        if 'aug_array.npy' in os.listdir('./data') and 'aug_label.csv' in os.listdir('./data'):
            train = np.load('./data/aug_array.npy')
            labels = pd.read_csv('./data/aug_label.csv')
            labels = labels['digit']
        else:
            train, labels = aug.data_with_aug()

        # data split
        X_train, X_val, y_train, y_val = aug.data_split(train, labels)
    else:
        X_train, X_val, y_train, y_val = dl.data_split()

    X_test = dl.test
    X_test = X_test.drop(['id', 'letter'], axis=1).values
    X_test = X_test.reshape(-1,28,28,1)
    X_test = X_test / 255

    # Model     
    model = create_cnn_model(X_train)
    # Model Compile 및 callback
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # scheduler, early_stop
    cosine_annealing = CosineAnnealingScheduler(T_max=100, eta_max=6e-4, eta_min=3e-5)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    batch_size = args.batch_size

    # Model 학습
        
    # epochs = args.epochs
    model.fit(X_train, y_train, epochs=1 ,validation_data=(X_val,y_val),
    shuffle=True, verbose=1, callbacks=[cosine_annealing, early_stop]
    )

    # submission
    submission = dl.submission
    submission['digit'] = np.argmax(model.predict(X_test), axis=1)
    submission.to_csv('./submissions/{}'.format(args.submission_name))

    if args.model_save == True:
        if os.path.isdir('./model') == False: os.mkdir('./model')
        model_name ='mnist_model_'
        model_num = str(np.random.randint(0, 9999999))
        model_fullname = model_name + model_num +'.h5'
        model.save('./model/{}'.format(model_fullname))
        print('Model Name : {} '.format(model_fullname))
