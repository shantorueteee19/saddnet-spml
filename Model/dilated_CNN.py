#Raw BVP path
#Dilated multi-scale CNN
def dilated_mscnn(input_layer):
  #path1
  x_mscnn1 = Conv1D(filters=128, kernel_size=3, strides=5, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
  x_mscnn1 = BatchNormalization()(x_mscnn1)
  x_mscnn1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_mscnn1)
  x_mscnn1 = ReLU()(x_mscnn1)
  x_mscnn1 = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(x_mscnn1)
  x_mscnn1 = BatchNormalization()(x_mscnn1)
  x_mscnn1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_mscnn1)
  #path2
  x_mscnn2 = Conv1D(filters=128, kernel_size=5, strides=5, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
  x_mscnn2 = BatchNormalization()(x_mscnn2)
  x_mscnn2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_mscnn2)
  x_mscnn2 = ReLU()(x_mscnn2)
  x_mscnn2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same',dilation_rate=2, kernel_regularizer=regularizers.l2(0.01))(x_mscnn2)
  x_mscnn2 = BatchNormalization()(x_mscnn2)
  x_mscnn2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_mscnn2)
  #path3
  x_mscnn3 = Conv1D(filters=128, kernel_size=9, strides=5, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
  x_mscnn3 = BatchNormalization()(x_mscnn3)
  x_mscnn3 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_mscnn3)
  x_mscnn3 = Conv1D(filters=32, kernel_size=9, strides=1, padding='same',dilation_rate=5, kernel_regularizer=regularizers.l2(0.01))(x_mscnn3)
  x_mscnn3 = BatchNormalization()(x_mscnn3)
  x_mscnn3 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_mscnn3)
  #path4
  x_mscnn4 = Conv1D(filters=128, kernel_size=13, strides=5, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
  x_mscnn = BatchNormalization()(x_mscnn4)
  x_mscnn4 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_mscnn4)
  x_mscnn4 = ReLU()(x_mscnn4)
  x_mscnn4 = Conv1D(filters=32, kernel_size=13, strides=1, padding='same',dilation_rate=7, kernel_regularizer=regularizers.l2(0.01))(x_mscnn4)
  x_mscnn4 = BatchNormalization()(x_mscnn4)
  x_mscnn4 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_mscnn4)
  #concatenate
  x = concatenate([x_mscnn1, x_mscnn2, x_mscnn3, x_mscnn4])
  return x