How to use

using vgg dict net dataset and network and this approach : https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5


# Reference

1 - To pre-train the CNN we use this dataset :

VGG dataset for words : http://www.robots.ox.ac.uk/~vgg/data/text/

2 - Use this project to generate images with many characters (not only text)

https://github.com/Belval/TextRecognitionDataGenerator

with this parameters :


python3 run.py -rs -rbl -rk -c 10000 -k 5 --output_dir out/train
python3 run.py -rs -rbl -rk -c 1000 -k 5 --output_dir out/test