import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, BatchNormalization, ReLU, LayerNormalization, Add, Multiply, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


#SADDnet model
def SADDNet_model(input_shape, num_class):
  #input layer
  input_layer = Input(shape=input_shape)
  #Raw path
  x_raw = raw_path(input_layer)
  #autoencoder
  x_encoded, x_decoded = autoencoder_decoder(input_layer)
  #Encoded path
  x_encoded = encoded_path(x_encoded)
  #Decoded path
  x_decoded = decoded_path(x_decoded)
  #Concat
  x_concat = concatenate([x_raw, x_encoded, x_decoded])
  #Classification
  x_concat = Conv1D(filters=96, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(x_concat)
  x_concat = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_concat)
  x_concat = BatchNormalization()(x_concat)
  x_concat = ReLU()(x_concat)
  x_concat = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(x_concat)
  x_concat = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_concat)
  x_concat = BatchNormalization()(x_concat)
  x_concat = ReLU()(x_concat)
  x_concat = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(x_concat)
  x_concat = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_concat)
  x_concat = BatchNormalization()(x_concat)
  x_concat = ReLU()(x_concat)
  #final classification
  x_gap = GlobalAveragePooling1D()(x_concat)
  x_out = Dense(num_class, activation='softmax')(x_gap)
  #model initialization
  model = Model(inputs=input_layer, outputs=x_out)

  #Model initialization
  model = Model(inputs=input_layer, outputs=x_out)
  return model