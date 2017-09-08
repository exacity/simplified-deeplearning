# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
    file name: main.py
    create time: 2017年09月01日 星期五 13时48分54秒
    author: Jipeng Huang
    e-mail: huangjipengnju@gmail.com
    github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
# based on tensorflow/example/tutorials deepdream
import os
import argparse
import tensorflow as tf
import numpy as np
import cv2

# parameter
model_name = "tensorflow_inception_graph.pb"
imagenet_mean = 117.0
layer = 'mixed4c'
iter_num = 50
octave_num = 4
octave_scale = 1.4
learning_rate = 1.5
tile_size = 512
noise = np.random.uniform(size=(224, 224, 3)) + 100.0


def define_args():
    """define args"""
    parser = argparse.ArgumentParser(description="deep_dream")
    parser.add_argument("-i", "--input", help="input path", default="none")
    parser.add_argument("-o", "--output", help="output path", default="output/output.jpg")
    return parser.parse_args()


def get_model():
    """download model"""
    model = os.path.join("model", model_name)
    if not os.path.exists(model):
        print("Down model...")
        os.system("wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -P model")
        os.system("unzip model/inception5h.zip -d model")
        os.system("rm model/inception5h.zip")
        os.system("rm model/imagenet_comp_graph_label_strings.txt")
    return model


def deep_dream(model, output_path, input_image=noise):
    """implement of deep dream"""
    # define graph
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    # load model
    with tf.gfile.FastGFile(model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # define input
    X = tf.placeholder(tf.float32, name="input")
    X2 = tf.expand_dims(X - imagenet_mean, 0)
    tf.import_graph_def(graph_def, {"input": X2})

    # L2 and gradient
    loss = tf.reduce_mean(tf.square(graph.get_tensor_by_name("import/%s:0" % layer)))
    gradient = tf.gradients(loss, X)[0]

    image = input_image
    octaves = []

    # tranforming TF function
    def tffunc(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    def resize(image, size):
        """resize image in nparray"""
        image = tf.expand_dims(image, 0)
        return tf.image.resize_bilinear(image, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)

    for i in range(octave_num - 1):
        size = np.shape(image)[:2]
        narrow_size = np.int32(np.float32(size) / octave_scale)
        # down sampling and up sampling equal to smooth, diff can save significance
        down = resize(image, narrow_size)
        diff = image - resize(down, size)
        image = down
        octaves.append(diff)

    def cal_gradient(image, gradient):
        """cal gradient"""
        # generate offset and shift to smooth tile edge
        shift_x, shift_y = np.random.randint(tile_size, size=2)
        image_shift = np.roll(np.roll(image, shift_x, 1), shift_y, 0)
        total_gradient = np.zeros_like(image)
        # calculate gradient for each region
        for y in range(0, max(image.shape[0] - tile_size // 2, tile_size), tile_size):
            for x in range(0, max(image.shape[1] - tile_size // 2, tile_size), tile_size):
                region = image_shift[y:y + tile_size, x:x + tile_size]
                total_gradient[y:y + tile_size, x:x + tile_size] = sess.run(gradient, {X: region})
        return np.roll(np.roll(total_gradient, -shift_x, 1), -shift_y, 0)

    for i in range(octave_num):
        print("octave num %s/%s..." % (i+1, octave_num))
        if i > 0:
            # restore image except original image
            diff = octaves[-i]
            image = resize(image, diff.shape[:2]) + diff
        for j in range(iter_num):
            # gradient ascent
            g_ = cal_gradient(image, gradient)
            image += g_ * (learning_rate / (np.abs(g_).mean() + 1e-7))  # large learning rate for small g_

    cv2.imwrite(output_path, image)


if __name__ == "__main__":
    args = define_args()
    model_path = get_model()
    if args.input == "none":
        deep_dream(model_path, args.output)
    else:
        image = np.float32(cv2.imread(args.input))
        deep_dream(model_path, args.output, input_image=image)
