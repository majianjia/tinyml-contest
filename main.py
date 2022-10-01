import matplotlib.pyplot as plt
# import keras.api._v2.keras as keras
# from tensorflow.keras import *
# from keras.datasets import mnist
# from keras.layers import *
# from keras.activations import *
# from keras.models import load_model, save_model, Model

from tensorflow.keras import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import load_model, save_model
import tensorflow as tf
from tensorflow.keras import backend as K

import pickle
import numpy as np

from nnom import *

# def precision(y_true, y_pred):
#     TP = tf.reduce_sum(y_true * tf.round(y_pred))
#     TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
#     FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
#     FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
#     return TP / (TP + FP + np.finfo(float).eps)
#
#
# def recall(y_true, y_pred):
#     TP = tf.reduce_sum(y_true * tf.round(y_pred))
#     TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
#     FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
#     FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
#     return TP / (TP + FN + np.finfo(float).eps)

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def fbscore(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    b = 2
    f = (1+b*b)*(prec*rec)/(b*b * prec + rec + K.epsilon())
    return f


def build_model(x_shape):
    inputs = Input(shape=x_shape)
    x = Conv1D(32, kernel_size=5, strides=1, padding='valid')(inputs)
    x = BatchNormalization()(x)

    x = Conv1D(64, kernel_size=5, strides=2, padding="valid")(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(4, strides=2, padding="same")(x)

    x = Conv1D(64, kernel_size=5, strides=2, padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(4, strides=2, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, kernel_size=5, strides=2, padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(4, strides=2, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)
    x = Activation("sigmoid")(x)
    # x = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    # x = GlobalAveragePooling2D()(x)
    #predictions = Softmax()(x)
    # predictions = Sigmoid()(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def build_model2(x_shape):
    inputs = Input(shape=x_shape)

    x = Conv1D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(2, strides=1, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(2, strides=1, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Conv1D(256, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(4, strides=2, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)
    # x = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    # x = GlobalAveragePooling2D()(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def build_model3(x_shape):
    inputs = Input(shape=x_shape)
    x1 = Conv1D(8, kernel_size=(5), strides=(2), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = MaxPooling1D(4)(x1)

    x2 = Conv1D(4, kernel_size=(9), strides=(2), padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Dropout(0.5)(x2)
    x2 = MaxPooling1D(4)(x2)

    x3 = Conv1D(4, kernel_size=(13), strides=(2), padding='same')(inputs)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Dropout(0.5)(x3)
    x3 = MaxPooling1D(4)(x3)
    x = concatenate([x1, x2, x3], axis=-1)

    x = Conv1D(4, kernel_size=(3), strides=(1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(2)(x)

    # x = BatchNormalization()(x)

    # you can use either of the format below.
    # x = RNN(SimpleRNNCell(16), return_sequences=True)(x)
    # x = SimpleRNN(16, return_sequences=True)(x)

    # x2 = RNN(LSTMCell(32), return_sequences=True)(x)
    # x1 = LSTM(32, return_sequences=True, go_backwards=True)(x)
    # x = concatenate([x1, x2], axis=-1)
    #
    # # Bidirectional with concatenate. (not working yet)
    # x1 = RNN(GRUCell(16), return_sequences=True)(x)
    # x2 = GRU(16, return_sequences=True, go_backwards=True)(x)
    # x = concatenate([x1, x2], axis=-1)


    # Bidirectional with concatenate. (not working yet)

    # x1 = LSTM(32, return_sequences=True)(x)
    # x2 = LSTM(32, return_sequences=True, go_backwards=True)(x)
    # x = add([x1, x2])
    x = GRU(8, return_sequences=True, dropout=0.4)(x)
    # x1 = GRU(32, return_sequences=True, dropout=0.2)(x)
    # x2 = GRU(32, return_sequences=True, go_backwards=True, dropout=0.2)(x)
    # x = concatenate([x1, x2], axis=-1)
    x = GRU(8, return_sequences=True, dropout=0.4)(x)
    

    # x = Conv1D(16, kernel_size=(3), strides=(1), padding='same')(x)

    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(1)(x)
    x = Activation("sigmoid")(x)
    #predictions = x #Softmax()(x)

    return Model(inputs=inputs, outputs=x)


def train(model, x_train, y_train, x_test, y_test, batch_size=256, epochs=50):
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', precision, recall, fbscore])

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy', precision, recall])
    model.summary()

    checkpoint_filepath = 'tmp'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='recall',
        mode='max',
        save_best_only=True,
        verbose=1)

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=True,
             callbacks = [model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)
    return history, model

def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    with open("train_data.pkg", "rb") as fp:
        train_data = pickle.load(fp)
    with open("train_label.pkg", "rb") as fp:
        train_label = pickle.load(fp)
    with open("test_data.pkg", "rb") as fp:
        test_data = pickle.load(fp)
    with open("test_label.pkg", "rb") as fp:
        test_label = pickle.load(fp)

    epochs = 20
    batch_size = 256
    timestamp_size = batch_size
    num_classes = 2

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    # shuffle
    idx = np.random.permutation(len(train_label))
    train_data, train_label = train_data[idx], train_label[idx]
    idx = np.random.permutation(len(test_label))
    test_data, test_label = test_data[idx], test_label[idx]

    # print("train label", np.average(train_label))
    # print("validation label", np.average(test_label))

    print(train_data.shape[0], 'train samples')
    print(test_data.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    # train_label = tf.keras.utils.to_categorical(train_label, num_classes)
    # test_label = tf.keras.utils.to_categorical(test_label, num_classes)

    # reshape to 4 d because we build for 4d?
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)

    train_data = np.clip(train_data, -1, 1)
    test_data = np.clip(test_data, -1, 1)
    print('train_data shape:', train_data.shape)

    # plt.hist(train_data.flatten(), bins=100)
    # plt.show()

    model = build_model3(test_data.shape[1:])

    history = None
    history, model = train(model, train_data, train_label, test_data, test_label, batch_size, epochs)
    model.save("model")

    del model
    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model('model', custom_objects={'precision': precision, 'recall':recall, 'fbscore':fbscore})

    generate_test_bin(test_data, test_label, name="test_data_with_label.bin")
    print("true label count", np.sum(test_label))

    generate_model(model, test_data[:2048], quantize_method='kld', name=r'weights_contest.h')
    #generate_model(model, test_data[:2048], per_channel_quant=True, name=r'weights_contest.h')


    if(history != None):
        plt.plot(history.history["accuracy"], label="accuracy")
        plt.plot(history.history["precision"], label="precision")
        plt.plot(history.history["recall"], label="recall")
        plt.plot(history.history["fbscore"], label="fbscore")

        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.plot(history.history["val_precision"], label="val_precision")
        plt.plot(history.history["val_recall"], label="val_recall")
        plt.plot(history.history["val_fbscore"], label="val_fbscore")
        plt.legend()
        plt.show()

    # --------- for test in CI ----------
    # evaluate in Keras (for comparision)
    import sys
    #scores = evaluate_model(model, test_data, test_label)
    # reload model.
    model = tf.keras.models.load_model('model', custom_objects={'precision': precision, 'recall': recall, 'fbscore': fbscore})
    scores = model.evaluate(test_data, test_label, verbose=2)


    print(scores)
    # build NNoM
    os.system("scons")

    # do inference using NNoM
    cmd = r".\nnom.exe" if 'win' in sys.platform else "./nnom"
    os.system(cmd)
    try:
        # get NNoM results
        import pandas as pd
        result = pd.read_csv("result.csv")
        # result = np.genfromtxt('result.csv', dtype=np.int, skip_header=1)
        # result = result[:, 0]  # the first column is the label, the second is the probability
        result = result["predic"].values
        label = test_label.flatten()  # use the original numerical label
        acc = np.sum(result == label).astype('float32') / len(result)
        if (acc > 0.5):
            print("Top 1 Accuracy on Tensorflow %.2f%%" % (scores[1] * 100))
            print("Top 1 Accuracy on NNoM  %.2f%%" % (acc * 100))
            return 0
        else:
            raise Exception('test failed, accuracy is %.1f%% < 80%%' % (acc * 100.0))
    except:
        raise Exception('could not perform the test with NNoM')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
