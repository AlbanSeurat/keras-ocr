import os
import sys
from models import ocr
from keras.optimizers import SGD
from keras.callbacks import LambdaCallback
import keras.backend as K
import numpy as np
from tensorflow.python import debug as tf_debug
from utils.datasets import CustomGeneratedDataSet, CnnRnnGenerator
from utils.swa import SWA
from utils.weights import WeightsDumper

image_dir = "../TextRecognitionDataGenerator/TextRecognitionDataGenerator/out"
dataset = CustomGeneratedDataSet(image_dir)


def decode_predict_ctc(out, top_paths=1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1],
                                          greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
        text = dataset.labels_to_text(lables)
        results.append(text)
    return results


if "debug" in sys.argv:
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

if "train" in sys.argv:
    ocrModel = ocr.OcrWithLoss(dataset.lexicon_len(), cnn_trainable=True)
    dumper = WeightsDumper(ocrModel)
    dumper.restore()

    swa = SWA("test", 8)

    optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    ocrModel.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer, metrics=['accuracy'])

    generator = CnnRnnGenerator(dataset)
    ocrModel.summary()

    ocrModel.fit_generator(generator=generator, epochs=10, callbacks=[swa, LambdaCallback(on_batch_end=lambda batch, logs: dumper.dump())], use_multiprocessing=True, workers=4)

elif "predict" in sys.argv:

    ocrModel = ocr.Ocr(dataset.lexicon_len(), weights=None)
    dumper = WeightsDumper(ocrModel)
    dumper.restore()

    for root, dirs, files in os.walk("output"):
        files = (x for x in files if x.endswith("png"))
        for filename in files:
            img = dataset.preprocess(filename, dir="output")
            predicted = ocrModel.predict(img)
            print(filename, decode_predict_ctc(predicted))
    
else:
    raise NotImplementedError
