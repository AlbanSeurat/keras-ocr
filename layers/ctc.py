from keras.layers import Layer, Input
import keras.backend as K


class CtcLoss(Layer):

    def __init__(self, inputs, **kwargs):
        self.labels, self.input_length, self.label_length = inputs
        super(CtcLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(CtcLoss, self).build(input_shape)  # Be sure to call this at the end

    def call(self, y_pred):
        return K.ctc_batch_cost(self.labels, y_pred, self.input_length, self.label_length)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


class CtcDecode(Layer):

    def __init__(self, input_length, **kwargs):
        self.input_length = input_length
        super(CtcDecode, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(CtcDecode, self).build(input_shape)  # Be sure to call this at the end

    def call(self, y_pred):
        top_k_decoded, logs = K.ctc_decode(y_pred,  K.reshape(self.input_length, (-1,)), greedy=True)
        return K.reshape(top_k_decoded,(-1,1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
