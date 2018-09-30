from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Conv2D, Input, Flatten, Dropout, MaxPooling2D


class ImageToChar:

    # Hyperparameters
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    def build(self, nb_classes, input_shape=(28, 28, 1), verbose=False):

        self.model = Sequential()
        self.model.add(Conv2D(self.nb_filters,
                                self.kernel_size,
                                padding='valid',
                                input_shape=input_shape,
                                activation='relu'))
        self.model.add(Conv2D(self.nb_filters,
                                self.kernel_size,
                                activation='relu'))

        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())

        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        if verbose == True:
            print(self.model.summary())
        return self.model

    def fit(self, X_train, Y_train, batch_size, nb_epoch):
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)


    def score(self, X_test, Y_test):
        score = self.model.evaluate(X_test, Y_test, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score


    def predict(self, X_test):
        # Predict the label for X_test
        return self.model.predict_classes(X_test)


    def save(self):
        # Offload model to file
        model_yaml = self.model.to_yaml()
        with open("bin/model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        save_model(self.model, 'bin/model.h5')

    def load(self):
        self.model = load_model('bin/model.h5')