import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from tqdm import tqdm

class Dataloader:
    def __init__(self):
        self.train = pd.read_csv(os.path.join('data', 'train.csv'))
        self.test = pd.read_csv(os.path.join('data', 'test.csv'))
        self.submission = pd.read_csv(os.path.join('data','submission.csv'))

    def make_train(self):
        train = self.train.drop(['id', 'digit', 'letter'], axis=1).values
        train = train.reshape(-1, 28, 28)

        label = self.train['digit']
        
        return train, label

    def data_split(self):
        x, y = self.make_train()
        x = x.reshape(-1, 28, 28, 1)
        x_train = x / 255

        y_train = np.zeros((len(y), len(y.unique())))
        for i, digit in enumerate(y):
            y_train[i, digit] = 1
        
        X_train, X_val , Y_train, Y_val = train_test_split(x_train, y_train, stratify=y_train, random_state=1234)
        return X_train, X_val, Y_train, Y_val

    def image_generator_local_save(self):
        for i in range(10):
            if os.path.isdir('./data/aug/{}/'.format(i)) != True:
                os.mkdir('./data/aug/{}/'.format(i))
        # dataset
        train, labels = self.make_train()
        print(train.shape)
        # Image generator
        image_generator = ImageDataGenerator(
            rotation_range = 45,
            width_shift_range = 0.20,
            height_shift_range = 0.20,
            zoom_range = 0.2,
            horizontal_flip=True,
            fill_mode = 'nearest',
        )
        # save at local
        for label, i in zip(labels ,range(train.shape[0])):
            num = 0
            for batch in image_generator.flow(train[i,:,:,np.newaxis].reshape((1,28,28,1)), batch_size=1, save_prefix=label , save_to_dir='./data/aug/{}/'.format(label), save_format='png'):
                num +=1
                if num >10:
                    break
        

class Augmented_dataset(Dataloader):
    def __init__(self):
        super(Augmented_dataset,self).__init__()
        self.aug_path = glob('./data/aug/*/*')
        self.aug_label_path = glob('./data/aug/*')
        
    def data_with_aug(self):
        '''
        img, label
        '''
        # data path
        paths = self.aug_path
        labels = self.aug_label_path
        label_list = []
        
        # 빈 array
        mnist_data = np.empty((0,28,28))
        # aug data 하나 씩 불러오기 
        # label 다시 짜야함
        for path in tqdm(paths):
            # label save
            label = path.split('/')[3]
            label_list.append(label)

            img = tf.io.read_file(path)
            img = tf.io.decode_png(img)
            img = np.squeeze(img)
            img = img.reshape((1,28,28))
            
            mnist_data = np.concatenate([mnist_data, img]).astype(np.float32)
        # original 호춯
        origin_train, origin_label  = self.make_train()
        # 기존 train 과 aug 데이터 결합
        total_label = list(origin_label) 
        mnist_data = np.concatenate([origin_train, mnist_data])
        
        total_label.extend(label_list)
        total_label = pd.Series(total_label, name='digit').astype(np.int32)

        np.save('./data/aug_array.npy', mnist_data)
        total_label.to_csv('./data/aug_label.csv',index=False)
        return mnist_data, total_label
    def data_split(self, images, labels):
        x, y = images, labels
        x = x.reshape(-1, 28, 28, 1)
        x_train = x / 255

        y_train = np.zeros((len(y), len(y.unique())))

        for i, digit in enumerate(y):
            y_train[i, digit] = 1
        
        X_train, X_val , Y_train, Y_val = train_test_split(x_train, y_train, stratify=y_train, random_state=1234)
        return X_train, X_val, Y_train, Y_val