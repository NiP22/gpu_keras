import os
from keras import utils
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers


class CoolerGenerator(utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __getitem__(self, item):
        first_index = self.batch_size*item
        if self.batch_size*item > len(self.x):
            last_index = len(self.x)
        else:
            last_index = self.batch_size*(item + 1)
        sample = self.x[first_index:last_index]
        sample = sample.reshape((sample.shape[0], sample.shape[1], 1))
        targets = self.y[first_index:last_index]
        return sample, targets

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x = sequence[i:end_ix]
        seq_y = sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


if __name__ == '__main__':


    f = open('jena_climate_2009_2016.csv')
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    print(header)
    print(len(lines))

    float_data = np.zeros((len(lines), len(header) - 1))
    print(len(float_data))

    X = np.zeros((len(lines), len(header) - 2))
    temperature = np.zeros((len(lines),))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
        X[i, :] = list([values[0]]) + values[2:]
        temperature[i] = values[1]
    n_steps = 3

    train_x, train_y = split_sequence(temperature[:350000], n_steps)
    val_x, val_y = split_sequence(temperature[350000:], n_steps)
    batch_size = 150
    train_generator = CoolerGenerator(train_x, train_y, batch_size)
    validation_generator = CoolerGenerator(val_x, val_y, batch_size)
    model = Sequential()
    model.add(layers.LSTM(32, input_shape=(n_steps, 1)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mae')
    history = model.fit_generator(generator=train_generator,
                                  epochs=20,
                                  validation_data=validation_generator,
                                  workers=10,
                                  use_multiprocessing=True)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label="Validation loss")
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

