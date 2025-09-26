def autoencoder_decoder(input_layer):
  #Encoder
  encode = Conv1D(filters=128, kernel_size=3, strides=5, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
  encode = ReLU()(encode)
  encode = MaxPooling1D(pool_size=2, strides=1, padding='same')(encode)
  encode = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(encode)
  encode = ReLU()(encode)
  encode = MaxPooling1D(pool_size=2, strides=1, padding='same')(encode)
  encode = Conv1D(filters=96, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(encode)
  encode = ReLU()(encode)
  encode = MaxPooling1D(pool_size=2, strides=1, padding='same')(encode)
  #Decoder
  decode = Conv1D(filters=96, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(encode)
  decode = ReLU()(decode)
  decode = MaxPooling1D(pool_size=2, strides=1, padding='same')(decode)
  decode = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(decode)
  decode = ReLU()(decode)
  decode = MaxPooling1D(pool_size=2, strides=1, padding='same')(decode)
  decode = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(decode)
  decode = ReLU()(decode)
  decode = MaxPooling1D(pool_size=2, strides=1, padding='same')(decode)
  decoded = Conv1D(filters=1, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(decode)
  #Decoded out
  # decoded = Activation('sigmoid')(decoded)
  decoded = LayerNormalization()(decoded)
  #We need encoded output
  return encode, decoded