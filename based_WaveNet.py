import keras as keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.layers import Conv1D, Input, Add, Activation, Dropout, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error



# Replace YOUR_STOCK_SYMBOL with the stock symbol you want to fetch
yfin.pdr_override()
df = pdr.get_data_yahoo('AAPL', start = '1980-12-12', end = '2022-12-17')
print(df.head)
df.shape

# define hyperparameters
n_filters = 128
filter_width = 2
dilation_rates = [2**i for i in range(10)] * 3
dropout_rate = 0.05
n_classes = 10
batch_size = 32
epochs = 10 #100

# Check for zero or negative values in closing prices
closing_prices = df['Close'].values
if np.any(closing_prices <= 0):
    raise ValueError("Closing prices cannot contain zero or negative values.")

# Calculate log-returns
df['log_returns'] = np.log(df['Close']) - np.log(df['Close'].shift(1))    # np.diff(np.log(df['Close'])) #
df.dropna(inplace=True)

def generate_sequences(data, sequence_length):
    input_sequences = []
    target_sequences = []

    for i in range(len(data) - sequence_length):
        input_seq = data[i:i + sequence_length]
        target_seq = data[i + 1:i + sequence_length + 1]  # Shift the target sequence by one

        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    return np.array(input_sequences), np.array(target_sequences)

# generate input and label sequences
seq_length = 75 #60 # to do 20-30, 50-100
x_train, y_train = generate_sequences(df['log_returns'].values[:int(0.8*len(df))], seq_length)
x_test, y_test = generate_sequences(df['log_returns'].values[int(0.8*len(df)):], seq_length)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1, 1))
print(x_train.shape)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1, 1))
print(x_test.shape)


# reshape y_train and y_test
y_train = y_train[:, :]
print(y_train.shape)
y_test = y_test[:, :]

# one-hot encode y_train and y_test
y_train = to_categorical(y_train, num_classes=n_classes)
y_test = to_categorical(y_test, num_classes=n_classes)

# define WaveNet model architecture
input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))
skip_connections = []
residual_outputs = []

# first convolutional layer
conv = Conv1D(filters=n_filters, kernel_size=filter_width, padding='same')(input_layer)
# conv = Conv1D(filters=1, kernel_size=filter_width, padding='same')(input_layer)

# define residual blocks with dilated convolutions
for dilation_rate in dilation_rates:
    # dilated convolution
    conv_filter = Conv1D(filters=n_filters, kernel_size=filter_width, padding='same',
                         dilation_rate=dilation_rate)(conv)
    conv_gate = Conv1D(filters=n_filters, kernel_size=filter_width, padding='same',
                       dilation_rate=dilation_rate)(conv)
    gate_activation = Activation('sigmoid')(conv_gate)
    output_activation = Activation('relu')(conv_filter)
    gated_activation = Multiply()([output_activation, gate_activation])
    # skip connection
    skip_out = Conv1D(filters=n_filters, kernel_size=1, padding='same')(gated_activation)
    skip_connections.append(skip_out)
    # residual connection
    conv_residual = Conv1D(filters=n_filters, kernel_size=1, padding='same')(gated_activation)
    residual_outputs.append(conv_residual)
    # update input to residual output
    # conv = Add()(conv_residual)
    conv = Add()([conv, conv_residual])

# sum the skip connections
sum_skips = Add()(skip_connections)
# final activations
final_activations = Activation('relu')(sum_skips)
final_activations = Dropout(dropout_rate)(final_activations)
final_activations = Conv1D(filters=n_filters, kernel_size=1, padding='same')(final_activations)
final_activations = Activation('relu')(final_activations)
final_activations = Dropout(dropout_rate)(final_activations)
# output_layer = Conv1D(filters=n_filters, kernel_size=1, padding='same', activation='softmax')(final_activations)
# output_layer = Conv1D(filters=n_classes, kernel_size=1, padding='same', activation='softmax')(final_activations)
output_layer = Conv1D(filters=n_classes, kernel_size=1, padding='same', activation='sigmoid')(final_activations)

# define and compile model
skip_outputs = []
for i, skip_connection in enumerate(skip_connections):
    skip_out = Conv1D(filters=n_classes, kernel_size=1, padding='same', name=f"skip_out_{i}")(skip_connection)
    skip_outputs.append(skip_out)

model_output = Add()(skip_outputs)
model = Model(inputs=input_layer, outputs=model_output)
adam_optimizer = Adam(lr=1e-4)
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train model
filepath = 'model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train model and save the best performing model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks_list)

# Load the best performing model
best_model = keras.models.load_model("model.h5")

input_seqs, target_seqs = generate_sequences(y_test, seq_length) #(test_data, sequence_length)
print (input_seqs.shape)

# Use the best model for prediction on new data
predicted_seqs = best_model.predict(x_test)

# Evaluate the best model on test data
test_loss, test_acc = best_model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Calculate the accuracy for each sequence
accuracies = []
for i in range(len(x_test)):
    target_seq = np.argmax(y_test[i], axis=1)
    predicted_seq = np.argmax(predicted_seqs[i], axis=1)

    accuracy = np.mean(target_seq == predicted_seq)
    accuracies.append(accuracy)

# Calculate the overall accuracy
overall_accuracy = np.mean(accuracies)
print('Overall accuracy:', overall_accuracy)

mae = mean_absolute_error(y_test.reshape(-1, n_classes), predicted_seqs.reshape(-1, n_classes))
rmse = np.sqrt(mean_squared_error(y_test.reshape(-1, n_classes), predicted_seqs.reshape(-1, n_classes)))


# mae = mean_absolute_error(y_test, predicted_seqs)
# rmse = np.sqrt(mean_squared_error(y_test, predicted_seqs))
print(f"MAE: {mae}, RMSE: {rmse}")


# Plot the accuracy for each sequence

plt.figure(figsize = (16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USD ($)', fontsize=18)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.show()
#


# Plot the accuracy for each sequence
plt.figure(figsize=(10, 6))
plt.plot(accuracies)
plt.title('Accuracy per sequence')
plt.xlabel('Sequence')
plt.ylabel('Accuracy')
plt.show()

# Plot the distribution of accuracies
plt.figure(figsize=(10, 6))
plt.hist(accuracies, bins=20)
plt.title('Accuracy distribution')
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.show()
