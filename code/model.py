#!/usr/bin/env python
# encoding: utf-8
from random import randint
import time
from data import load_data
import tensorflow as tf
import numpy as np
from tools import extract_axis_1
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
"""
加载数据
"""
train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_data()
TRAIN_DATA_SIZE = len(train_data)
VALID_DATA_SIZE = len(valid_data)
TEST_DATA_SIZE = len(test_data)
wordVectors = np.load('GloVe/wordVectors.npy')
temp_words_list = np.load('GloVe/wordsList.npy')  # 字典
temp_words_list = temp_words_list.tolist()  # Originally loaded as numpy array
words_list = [word.decode('UTF-8') for word in temp_words_list]  # Encode words as UTF-8

"""
定义模型的超参数
"""
max_seq_length = 100
hidden_size = 128
class_num = 4
batch_size = 50
iterations = 5000


def _embedding_one_sentence(sentence):
    import re
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    cleaned_sentence = re.sub(strip_special_chars, "", sentence.lower())
    data_index_type = np.zeros(max_seq_length, dtype='int32')
    split = cleaned_sentence.split()
    word_count = 0
    for word in split:
        try:
            data_index_type[word_count] = words_list.index(word)
        except ValueError:
            data_index_type[word_count] = 399999  # 对于字典中没有的单词
        word_count += 1
        if word_count >= max_seq_length:
            break
    return data_index_type, word_count


"""
准备好一个batch的数据
"""


def generate_a_batch_data(train_batch_data, train_batch_labels):
    batch_data = train_batch_data[:batch_size]
    batch_label = train_batch_labels[:batch_size]
    """
    首先是main里面的batch数据
    """
    main_input_one_batch_data = np.zeros(dtype=np.int32, shape=(batch_size, max_seq_length))
    main_input_one_batch_length = np.zeros(dtype=np.int32, shape=(batch_size,))
    batch_one_labels = np.zeros(dtype=np.int32, shape=(batch_size, class_num))
    for i in range(batch_size):
        main_input_one_batch_data[i], main_input_one_batch_length[i] = _embedding_one_sentence(batch_data[i])
        batch_one_labels[i][batch_label[i]] = 1
    return main_input_one_batch_data, batch_one_labels, main_input_one_batch_length


"""
首先是主RNN模型
"""
with tf.variable_scope('main'):
    main_input = tf.placeholder(dtype=tf.int32, shape=(None, max_seq_length))
    main_sequence_length = tf.placeholder(dtype=tf.int32, shape=(None,))
    main_X = tf.nn.embedding_lookup(wordVectors, main_input)
    main_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    main_cell = tf.contrib.rnn.DropoutWrapper(main_cell, output_keep_prob=0.8)
    main_outputs, _ = tf.nn.dynamic_rnn(cell=main_cell, inputs=main_X, sequence_length=main_sequence_length,
                                        dtype=tf.float32)
    main_h_state = extract_axis_1(main_outputs, main_sequence_length - 1)

"""
然后是分类器部分
"""
with tf.variable_scope('merge'):
    # Define weights
    labels = tf.placeholder(tf.float32, [None, class_num])
    W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
    y_pre = tf.matmul(main_h_state, W) + bias
    correctPred = tf.equal(tf.argmax(y_pre, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pre, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

"""
计算模型
"""

# run model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session(config=config)


def train_model():
    global batch_size
    """
    准备好文件夹
    """
    logdir = "tensorboard/normal_{}/".format(int(time.time()))
    modeldir = "models/normal_models/"
    import shutil
    import os
    try:
        shutil.rmtree(logdir)
    except:
        pass
    try:
        shutil.rmtree(modeldir)
    except:
        pass
    os.mkdir(logdir)
    os.mkdir(modeldir)
    sess.run(tf.global_variables_initializer())
    accuracy_summary = tf.summary.scalar('Accuracy', accuracy)
    loss_summary = tf.summary.scalar('Loss', loss)
    writer = tf.summary.FileWriter(logdir, sess.graph)
    max_accuracy = 0.0
    print('RNN MODEL NOW')
    for i in range(iterations):
        random_num = randint(1, TRAIN_DATA_SIZE - batch_size)
        main_input_batch_data, batch_labels, main_input_batch_length = generate_a_batch_data(
            train_data[random_num:], train_labels[random_num:])
        _, loss_temp, current_loss_summary = sess.run([optimizer, loss, loss_summary], {
            main_input: main_input_batch_data,
            main_sequence_length: main_input_batch_length,
            labels: batch_labels
        })
        print('step', i, 'loss', loss_temp)
        if (i != 0 and i % 5 == 0) or i == 1:
            batch_size = 1000
            random_num = randint(1, VALID_DATA_SIZE - batch_size)
            main_input_batch_data, batch_labels, main_input_batch_length, = generate_a_batch_data(
                valid_data[random_num:], valid_labels[random_num:])
            accuracy_temp, current_accuracy_summary = sess.run([accuracy, accuracy_summary], {
                main_input: main_input_batch_data,
                main_sequence_length: main_input_batch_length,
                labels: batch_labels
            })
            writer.add_summary(current_loss_summary, global_step=i)
            writer.add_summary(current_accuracy_summary, global_step=i)
            print('step:', i, 'accuracy:', accuracy_temp, 'loss:', loss_temp)
            if accuracy_temp > max_accuracy:
                save_path = saver.save(sess, "models/normal_models/{}.ckpt".format(accuracy_temp),
                                       global_step=i)
                print("saved to %s" % save_path)
                max_accuracy = accuracy_temp
            batch_size = 50
    writer.close()


"""
测试模型
"""


def test_model():
    print('START TEST')
    model_file = tf.train.latest_checkpoint('models/normal_models/')
    saver.restore(sess, model_file)
    accuracy_all = 0.0
    for iter_time in range(150):
        random_num = iter_time * 50
        main_input_batch_data, batch_labels, main_input_batch_length, = generate_a_batch_data(
            test_data[random_num:], test_labels[random_num:])
        accuracy_temp = sess.run(accuracy, {
            main_input: main_input_batch_data,
            main_sequence_length: main_input_batch_length,
            labels: batch_labels
        })
        print('iter time {} , accuracy {}'.format(iter_time, accuracy_temp))
        accuracy_all += accuracy_temp
    accuracy_all /= 150
    print('Test Result: {}'.format(accuracy_all))


if __name__ == "__main__":
    train_model()
    test_model()
