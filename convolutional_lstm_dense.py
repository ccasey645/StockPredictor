import tensorflow as tf
import numpy as np
import traceback
import matplotlib.pyplot as plt
import csv
import sys

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def get_dataset(dataset, split_index, reverse=False):
    """
    split one dataset in to two for training and test.
    :param dataset:
    :param split_index:
    :return: train_dataset, test_dataset
    """
    if reverse:
        return dataset[split_index:], dataset[:split_index]
    else:
        return dataset[:split_index], dataset[split_index:]

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def fit_model(model, train_set, validation_data):
    history = model.fit(train_set, validation_data=validation_data, epochs=500)
    return history

def test_learning_rate(learning_rate, train_set, test_set):
    """
    Show chart displaying slope of learning rate of model. Use this to decide what learning rate
    to use on actual model.fit() run. Need to chose x value where slope is steady and decreasing.
    :param epoch: Number
    :param learning_rate:
    :param train_set:
    :param test_set:
    :return:
    """
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, 1]),
      tf.keras.layers.LSTM(256, return_sequences=True),
      tf.keras.layers.LSTM(256, return_sequences=True),
      tf.keras.layers.Dense(60, activation="relu"),
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(1),
      #tf.keras.layers.Lambda(lambda x: x * 400)
    ])

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: learning_rate * 10**(epoch / 20))
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(train_set, validation_data=test_set, epochs=100, callbacks=[lr_schedule])
    print(history.history)
    try:
      print("Printing learning rate!!!")
      print(history.history["lr"])
    except:
      print("Failed to print learning rate!")
      traceback.print_exc()

    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([0.01, 1, 0, 10000])
    plt.show()


def chart_results(history, rnn_forecast):
    # #-----------------------------------------------------------
    # # Retrieve a list of list results on training and test data
    # # sets for each training epoch
    # #-----------------------------------------------------------
    loss = history.history['loss']

    epochs = range(len(loss))  # Get number of epochs

    # #------------------------------------------------
    # # Plot training and validation loss per epoch
    # #------------------------------------------------
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss"])

    plt.figure()

    zoomed_loss = loss[200:]
    zoomed_epochs = range(200, 500)

    # #------------------------------------------------
    # # Plot training and validation loss per epoch
    # #------------------------------------------------
    plt.plot(zoomed_epochs, zoomed_loss, 'r')
    plt.title('Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss"])

    plt.figure()

    print(rnn_forecast)


def fit(model, train_set, test_set, time_valid, series, window_size, split_time):
    history = fit_model(model, train_set, test_set)
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, test_set)
    plot_series(time_valid, rnn_forecast)

    tf.keras.metrics.mean_absolute_error(test_set, rnn_forecast).numpy()
    chart_results(history, rnn_forecast)

def run(determine_learning_rate=False):
    time_step = []
    closing_prices = []

    # with open('/tmp/sunspots.csv') as csvfile:
    with open('Data/dji_history.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        skip_count = 0
        for index, row in enumerate(reader):
            # if skip_count % 5 == 0:
            #     skip_count = 0
            #     print(row[4])
            closing_prices.append(float(row[4]))
            time_step.append(index)
            # else:
            #     skip_count += 1

    series = np.array(closing_prices)
    time = np.array(time_step)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)

    series = np.array(closing_prices)
    time = np.array(time_step)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)

    shuffle_buffer_size = 400
    window_size = 30
    batch_size = 120
    learning_rate = 9.5e-3

    print("Printing length!")
    print(len(series))

    split_time = 4400
    time_train = time[:split_time]
    # x_train = series[:split_time]
    time_valid = time[split_time:]
    # x_valid = series[split_time:]

    tf.keras.backend.clear_session()
    x_train, x_valid = get_dataset(series, split_time)
    tf.random.set_seed(51)
    np.random.seed(51)

    train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    test_set = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)

    print(train_set)
    print(x_train.shape)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, 1]),
      tf.keras.layers.LSTM(512, return_sequences=True),
      tf.keras.layers.LSTM(512, return_sequences=True),
      tf.keras.layers.Dense(240, activation="relu"),
      tf.keras.layers.Dense(80, activation="relu"),
      tf.keras.layers.Dense(1),
      #tf.keras.layers.Lambda(lambda x: x * 400)
    ])

    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    if determine_learning_rate:
        learning_rate = 0.01
        test_learning_rate(learning_rate, train_set, test_set)
    else:
        fit(model, train_set, test_set, time_valid, series, window_size, split_time)
        #Now, reverse the train and test data and rerun it so we can train on all the data
        x_train, x_valid = get_dataset(series, split_time, reverse=True)
        train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
        test_set = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)
        fit(model, train_set, test_set, time_valid, series, window_size, split_time)

if __name__ == "__main__":
    determine_learning_rate = False
    if "--get-learning-rate" in sys.argv:
        determine_learning_rate = True
    run(determine_learning_rate=determine_learning_rate)

