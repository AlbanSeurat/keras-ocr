from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout, Input
from utils.mat import MatFile
from numpy import array, dot, float32, mean, std, zeros, argmax
from skimage.transform import resize


class DictNet(Model):

    def __init__(self, weights=None, lex=None, trainable=True):

        mat = MatFile()

        input = x = Input(shape=(32, 100, 1), name="dict_input")

        # layer 1
        x = Conv2D(filters=64, kernel_size=(5, 5), input_shape=(32, 100, 1), padding='same', activation='relu',
                   name='layer1')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # layer 2
        x = Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', name="layer2")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # layer 3
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name="layer3")(x)

        # layer 3.5
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name="layer3b")(x)
        x = ZeroPadding2D(padding=((0, 0), (0, 1)))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # layer 4
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name="layer4")(x)

        # layer 5 (dense 1)
        x = Flatten()(x)
        x = Dense(4096, activation='relu', name="layer5")(x)
        x = Dropout(0.5)(x)

        # layer 6 (dense 2)
        x = Dense(4096, activation='relu', name="layer6")(x)
        x = Dropout(0.5)(x)

        # output layer
        output = Dense(88172, activation='softmax', name="output")(x)

        if lex is not None:
            data = mat.load(lex)
            self.output_word = data['lexicon']

        super(DictNet, self).__init__(inputs=input, outputs=output, name="DictNet")

        if weights is not None:
            data = mat.load(weights)

            mat.load_layer(data['layers'][0], self.get_layer(name="layer1"), shape=(5, 5, 1, 64))
            mat.load_layer(data['layers'][3], self.get_layer(name="layer2"))
            mat.load_layer(data['layers'][6], self.get_layer(name="layer3"))
            mat.load_layer(data['layers'][8], self.get_layer(name="layer3b"))
            mat.load_layer(data['layers'][11], self.get_layer(name="layer4"))
            mat.load_layer(data['layers'][13], self.get_layer(name="layer5"), shape=(26624, 4096))
            mat.load_layer(data['layers'][15], self.get_layer(name="layer6"), shape=(4096, 4096))
            mat.load_layer(data['layers'][17], self.get_layer(name="output"), shape=(4096, 88172))

        if not trainable:
            for layer in self.layers:
                layer.trainable = False

    @staticmethod
    def _rgb2gray(rgb):
        return dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def preprocess(img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = DictNet._rgb2gray(img)
        img = resize(img, (32, 100), order=1, preserve_range=True)
        img = array(img, dtype=float32)  # convert to single precision
        img = (img - mean(img)) / (std(img) + 0.0001)

        result = zeros((1, 32, 100, 1))
        result[0, :, :, 0] = img

        return result

    def classify_image(self, img):
        img = DictNet.preprocess(img)
        y_pred = self.predict(img, verbose=0)[0]
        y_pred = argmax(y_pred)
        return y_pred, self.output_word[y_pred]
