from keras.models import Sequential

model = Sequential()
model.summary()
for layer in model.layers:
    # model.get_layer('name') or model.get_layer(index==0)
    # model.layers[0:5] or model.layers[0]
    print("input" + str(layer.input_shape)+"output" + str(layer.output_shape))
