#Decoded path
def separable_fire(input_layer):
  #Squeeze
  x_initial = SeparableConv1D(filters=128, kernel_size=1, strides=1, padding='same')(input_layer)
  x_initial = ReLU()(x_initial)
  #Expand
  x_expand1 = SeparableConv1D(filters=32, kernel_size=1, strides=1, padding='same')(x_initial)
  x_expand1 = ReLU()(x_expand1)
  x_expand2 = SeparableConv1D(filters=32, kernel_size=3, strides=1, padding='same')(x_initial)
  x_expand2 = ReLU()(x_expand2)
  #Concat
  x_out = concatenate([x_expand1, x_expand2])
  return x_out
#Decoded
def decoded_path(input_layer):
  #Decoder out
  # x_decoded = auto_encoder(input_layer)[1] #decoded output
  x_decoded = input_layer
  #Fire
  x_fire = separable_fire(x_decoded)
  x_fire = separable_fire(x_fire)
  x_fire = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_fire)
  x_fire = separable_fire(x_fire)
  x_fire = separable_fire(x_fire)
  x_fire = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_fire)
  x_fire = separable_fire(x_fire)
  x_fire = separable_fire(x_fire)
  x_out = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_fire)
  return x_out