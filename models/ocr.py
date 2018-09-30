from models import dict
from keras.models import Model
from keras.layers import GRU, Reshape, Input, Dense, Activation
from keras.layers.merge import add, concatenate
from layers.ctc import CtcLoss
import keras.backend as K


class _OcrBase(Model):

    rnn_size = 64

    def _relu6(self, x):
        return K.relu(x, max_value=6)

    def __init__(self, lexicon_size, weights="data/dictnet.mat"):
        parent = dict.DictNet(weights=weights, trainable=False)

        layer4_output = parent.get_layer("layer4").output

        x = Reshape(target_shape=(52, 512))(layer4_output)

        gru_1 = GRU(self.rnn_size, return_sequences=True,
                    kernel_initializer='he_normal', name='gru1')(x)

        gru_1b = GRU(self.rnn_size, return_sequences=True,
                     go_backwards=True, kernel_initializer='he_normal',
                     name='gru1_b')(x)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(self.rnn_size, return_sequences=True,
                    kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(self.rnn_size, return_sequences=True, go_backwards=True,
                     kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

        gru_concat = concatenate([gru_2, gru_2b])

        # transforms RNN output to character activations:
        inner = Dense(lexicon_size, kernel_initializer='he_normal',
                      name='dense2')(gru_concat)
        y_pred = Activation('softmax', name='softmax')(inner)

        super(_OcrBase, self).__init__(inputs=parent.input, outputs=y_pred, name="_OcrBase")


class OcrWithLoss(Model):

    def __init__(self, lexicon_size, weights="data/dictnet.mat"):
        ocr = _OcrBase(lexicon_size, weights)

        input_length = Input(name='ctc_input_length', shape=[1], dtype='int64')
        label_length = Input(name='ctc_label_length', shape=[1], dtype='int64')
        labels = Input(name='ctc_labels', shape=(32,), dtype='float32')

        loss = CtcLoss(inputs=[labels, input_length, label_length], name="ctc")(ocr.output)

        super(OcrWithLoss, self).__init__(inputs=[ocr.input, input_length, labels, label_length], outputs=loss,
                                          name="OcrLoss")


class Ocr(Model):

    def __init__(self, lexicon_size, weights="data/dictnet.mat"):
        ocr = _OcrBase(lexicon_size, weights)

        #input_length = Input(name='ctc_input_length', shape=[1], dtype='int64')
        #decode = CtcDecode(input_length, name="ctc")(ocr.output)

        #super(Ocr, self).__init__(inputs=[ocr.input, input_length], outputs=decode, name="Ocr")
        super(Ocr, self).__init__(inputs=ocr.input, outputs=ocr.output, name="Ocr")
