from keras.utils import Sequence
import matplotlib.image as mpimg
from models.dict import DictNet
import numpy as np


class DictNetDataSet:

    def __init__(self, dir):
        self.dir = dir

        with open("%s/lexicon.txt" % dir) as f:
            self.labels = [line.strip() for line in f.readlines()]
            self.chars = sorted(set("".join(self.labels)))
            self.chars.append(' ')

    def images(self, gen_type):
        with open("%s/annotation_%s.txt" % (self.dir, gen_type)) as f:
            return f.readlines()

    def preprocess(self, image_name, dir=None):
        return DictNet.preprocess(mpimg.imread("%s/%s" % (self.dir if dir is None else dir, image_name)))

    def lexicon_len(self):
        return len(self.chars)

    def labels_from_id(self, id : int):
        return self.text_to_labels(self.labels[int(id)])

    # Translation of characters to unique integer values
    def text_to_labels(self, text):
        ret = []
        for char in text:
            ret.append(self.chars.index(char))
        return ret

    # Reverse translation of numerical classes back to characters
    def labels_to_text(self, labels):
        ret = []
        for c in labels:
            if c == len(self.chars):  # CTC Blank
                ret.append("")
            else:
                ret.append(self.chars[c])
        return "".join(ret)


class DictNetGenerator(Sequence):
    maxTextLen = 32

    def __init__(self, dataset : DictNetDataSet, batch_size=50, gen_type="train", shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.images = dataset.images(gen_type)

        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_tmp_images = [self.images[k] for k in indexes]

        X = np.empty((self.batch_size, 32, 100, 1))

        labels = np.ones((self.batch_size, self.maxTextLen), dtype=int) * -1
        input_length = np.zeros([self.batch_size, 1])
        label_length = np.zeros([self.batch_size, 1])

        for i, image in enumerate(list_tmp_images):
            image_name, id = map(str.strip, image.split(" "))
            X[i,] = self.dataset.preprocess(image_name)
            label = np.asarray(self.dataset.labels_from_id(id))
            # label = np.asarray([self.chars.index(c) for c in self.labels[int(id)]])
            labels[i, 0:label.shape[0]] = label
            label_length[i] = len(label)
            input_length[i] = self.batch_size

        inputs = {
            'dict_input': X,
            'ctc_labels': labels,
            'ctc_input_length': input_length,
            'ctc_label_length': label_length
        }

        outputs = {'ctc': np.zeros([self.batch_size])}

        return (inputs, outputs)

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)
