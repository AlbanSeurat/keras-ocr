from models import dict
import matplotlib.image as mpimg
import os

model = dict.DictNet(weights= "data/dictnet.mat", lex = "data/lex.mat")

for root, dirs, files in os.walk("output"):

    files = (x for x in files if x.endswith("png"))

    for filename in files:
        img = mpimg.imread("output/%s" % filename)
        print(filename, model.classify_image(img))