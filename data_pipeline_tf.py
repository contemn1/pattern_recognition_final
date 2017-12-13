import tensorflow as tf
import IOUtil
import tensorflow.contrib.eager as tfe
import numpy as np


def string_transformer(glove_dict):
    def sentence_to_vector(sentence):
        sentence = sentence.decode("utf-8")
        sents = sentence.strip().split(" ")
        gloves = [glove_dict[word] for word in sents if word in glove_dict]
        data_np = np.array(gloves, dtype=np.float32)
        return data_np

    return sentence_to_vector



if __name__ == '__main__':

    DATA_PATH = "/Users/zxj/PycharmProjects/pattern_recognition_final/"
    glove_dict = IOUtil.get_glove_nips("/Users/zxj/Downloads/nips_data/glove_nips.txt")
    dataset = tf.data.TextLineDataset(DATA_PATH + "nips_valid_sorted.txt")
    dataset = dataset.map(
        lambda text: tf.py_func(string_transformer(glove_dict), inp=[text], Tout=tf.float32))
    dataset = dataset.padded_batch(128, padded_shapes=[None, None])

    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(4):
            res = sess.run(one_element)
            print(res)
