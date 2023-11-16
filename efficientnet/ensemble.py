from keras.models import load_model
from keras.layers import Input, concatenate, Dense
from keras.models import Model
from keras.utils import plot_model
import numpy as np

# Load saved models
model1 = load_model('fightnight_iter1.h5')
model2 = load_model('fightnight_iter1.h5')

# Ensure that all models have the same input shape
input_shape = model1.input_shape[1:]

# Create an ensemble model
def define_ensemble_model(models, input_shape):
    # Ensure all models have the same input shape
    for model in models:
        assert model.input_shape[1:] == input_shape, "Input shapes of models must match."

    # Explicitly set the name of the input layers
    model1.get_layer('rescaling_2')._name = 'rescaling_2_input_model1'
    model2.get_layer('rescaling_2')._name = 'rescaling_2_input_model2'
    n = 0
    # Make all layers in all models not trainable
    for i, model in enumerate(models):
        for layer in model.layers:
            n += 1
            layer.trainable = False
            layer._name = 'ensemble_model_' + str(n) + '_' + layer.name

    # Define multi-headed input
    ensemble_visible = [model.input for model in models]

    # Concatenate the outputs of the models
    ensemble_outputs = [model.output for model in models]
    merge = concatenate(ensemble_outputs)

    # Dense layer for binary classification
    dense = Dense(1, activation='sigmoid', name='ensemble_output')(merge)

    # Create and compile the ensemble model
    ensemble_model = Model(inputs=ensemble_visible, outputs=dense)
    ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return ensemble_model

# Define the ensemble model
ensemble_model = define_ensemble_model([model1, model2], input_shape)

# Visualize the ensemble model
plot_model(ensemble_model, show_shapes=True, to_file='./ensemble_model.png')
