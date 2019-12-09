# conda install -c conda-forge tensorflow
import tensorflow.keras as k
import tensorflow as tf
# import tensorflow
import src.misc_func as aux
import numpy as np



def init():
    """
    default model, always overfits
    """
    model = k.Sequential()
    model.add(k.layers.LSTM(units=1000,
                            input_shape=(1, 16000),
                            return_sequences=True,
                            ))
    model.add(k.layers.Dropout(0.2))
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.LSTM(units=700,
                            return_sequences=True,
                            ))
    model.add(k.layers.LSTM(units=500,
                            return_sequences=True
                            ))
    model.add(k.layers.Dropout(0.2))
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Activation('sigmoid'))
    model.add(k.layers.Dropout(0.2))
    model.add(k.layers.TimeDistributed(k.layers.Dense(units=300)))
    model.add(k.layers.Dense(units=150, activation='sigmoid'))
    model.add(k.layers.Dense(units=21, activation='softmax'))
    model.add(k.layers.Flatten())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def init2():
    """
    parallel model, takes too much hardware resources
    """
    model = k.Sequential()
    model.add(k.layers.LSTM(units=1,
                            input_shape=(16000, 1),
                            return_sequences=False,
                            ))
    # model.add(k.layers.LSTM(units=200,
    #                         return_sequences=True,
    #                         ))
    # model.add(k.layers.LSTM(units=300,
    #                         return_sequences=True
    #                         ))
    # model.add(k.layers.BatchNormalization())
    # model.add(k.layers.Activation('sigmoid'))
    # model.add(k.layers.Dropout(0.2))
    # model.add(k.layers.TimeDistributed(k.layers.Dense(units=300)))
    # model.add(k.layers.TimeDistributed(k.layers.Dense(units=50)))
    # model.add(k.layers.Dense(units=30, activation='sigmoid'))
    # model.add(k.layers.Dense(units=21, activation='softmax'))
    # model.add(k.layers.LSTM(units=21,
    #                         return_sequences=False
    #                         ))
    model.add(k.layers.Dense(units=21, activation='softmax'))

    # model.add(k.layers.Flatten())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def init3():
    """
    balanced stateful model with 4 batches per data
    """
    model = k.Sequential()
    model.add(k.layers.LSTM(units=1,
                            # input_shape=(4000, 1),
                            stateful=True,
                            return_sequences=False,
                            batch_input_shape=(400, 4000, 1)
                            ))
    # model.add(k.layers.LSTM(units=200,
    #                         return_sequences=False,
    #                         ))
    # model.add(k.layers.LSTM(units=300,
    #                         return_sequences=True
    #                         ))
    # model.add(k.layers.BatchNormalization())
    # model.add(k.layers.Activation('sigmoid'))
    # model.add(k.layers.Dropout(0.2))
    # model.add(k.layers.TimeDistributed(k.layers.Dense(units=300)))
    # model.add(k.layers.TimeDistributed(k.layers.Dense(units=50)))
    # model.add(k.layers.Dense(units=30, activation='sigmoid'))
    # model.add(k.layers.Dense(units=21, activation='softmax'))
    # model.add(k.layers.LSTM(units=21,
    #                         return_sequences=False
    #                         ))
    model.add(k.layers.Dense(units=21, activation='softmax'))

    # model.add(k.layers.Flatten())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train(model):
    data, labels = aux.read_data_from_csv()
    # data = np.reshape(data, (data.shape[0]/2, 2, data.shape[1]))  # data.shape[0]=2000, data.shape[1]=16000
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))  # data.shape[0]=2000, data.shape[1]=16000
    labels = k.utils.to_categorical(labels)

    model.fit(data, labels, epochs=5, batch_size=100, verbose=1)
    return model
    # model.predict(x='')


def train_on_batch(model, batch_num=0, batch_size=1000):
    for i in range(int(batch_num/2)):
        print("Starting", i, "/", batch_num)
        data, labels = aux.read_batch(batch_size, i)  # returns list of batch_size elements
        data, labels = aux.prepare_dataset(data, labels)
        model.fit(data, labels, epochs=7, batch_size=200, verbose=1)  # , validation_split=0.01
    for i in range(int(batch_num/2), batch_num):
        print("Starting", i, "/", batch_num)
        data, labels = aux.read_batch(batch_size, i)  # returns list of batch_size elements
        data, labels = aux.prepare_dataset(data, labels)
        model.fit(data, labels, epochs=3, batch_size=50, verbose=1, validation_split=0.01)
    return model


def save_model(model, path="paralell_model.json"):
    model_json = model.to_json()
    with open(path, "w+") as json_file:
        json_file.write(model_json)


def load_model(path="paralell_model.json"):
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return tf.keras.models.model_from_json(loaded_model_json)


def test_model(model, path):
    data = aux.read_single_data(path)
    return model.predict(data)


if __name__ == "__main__":
    model = init()
    model = train_on_batch(model, batch_num=50)
    save_model(model)
    while True:
        path = input("../data (0 za izlaz): ")
        if path == '0':
            exit(0)
        print(test_model(model, "../data"+path))
        print("prepoznata rec je:", aux.inverted_labels[np.argmax(test_model(model, "../data"+path))])


# /train/audio/one/d4d898d7_nohash_2.wav
