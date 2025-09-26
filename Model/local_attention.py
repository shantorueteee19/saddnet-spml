#local attention block
def local_attention_module(input_layer):
  x_input = input_layer
  #Conv
  x_conv = Conv1D(filters=96, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
  x_conv = BatchNormalization()(x_conv)
  x_conv = ReLU()(x_conv)
  #Multiply
  x_mul = Multiply()([x_input, x_conv])
  #Add
  x_out = Add()([x_input, x_mul])
  return x_out