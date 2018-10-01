from keras.utils import Sequence
import matplotlib.image as mpimg
from models.dict import DictNet
import numpy as np


class _BaseDataSet:

    def __init__(self, dir, chars, labels, images):
        self.dir = dir
        self.chars = chars
        self.labels = labels
        self.images = images

    def preprocess(self, image_name, dir=None):
        return DictNet.preprocess(mpimg.imread("%s/%s" % (self.dir if dir is None else dir, image_name)))

    def lexicon_len(self):
        return len(self.chars)

    def labels_from_id(self, id: int):
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


class DictNetDataSet(_BaseDataSet):

    def __init__(self, dir):
        with open("%s/lexicon.txt" % dir) as f:
            labels = [line.strip() for line in f.readlines()]
            chars = sorted(set("".join(labels)))
            chars.append(' ')

        def splitter(line):
            splited = line.split(" ")
            return (splited[0], splited[1])

        with open("%s/annotation_%s.txt" % (self.dir, "train")) as f:
            images = [splitter(line) for line in f.readlines()]

        super(DictNetDataSet, self).__init__(dir, chars, labels, images)


class CustomGeneratedDataSet(_BaseDataSet):

    def __init__(self, dir):
        with open("%s/train/labels.txt" % dir) as f:
            lines = [line.strip().split(" ") for line in f.readlines()]
            labels = [line[1] for line in lines]
            chars = sorted(set("".join(labels)))
            chars.append(' ')

            images = [("train/%s" % line[0], idx) for (idx, line) in enumerate(lines)]

        super(CustomGeneratedDataSet, self).__init__(dir, chars, labels, images)


class DataSetGenerator(Sequence):
    maxTextLen = 32

    def __init__(self, dataset: DictNetDataSet, batch_size=50, gen_type="train", shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.images = dataset.images.copy()

        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_tmp_images = [self.images[k] for k in indexes]

        X = np.empty((self.batch_size, 32, 100, 1))

        labels = np.ones((self.batch_size, self.maxTextLen), dtype=int) * -1
        input_length = np.zeros([self.batch_size, 1])
        label_length = np.zeros([self.batch_size, 1])

        for i, image in enumerate(list_tmp_images):
            image_name, id = image
            X[i,] = self.dataset.preprocess(image_name)
            label = np.asarray(self.dataset.labels_from_id(id))
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
