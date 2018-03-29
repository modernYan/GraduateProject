#!/usr/bin/env python
# encoding: utf-8
from random import randint

import time

from data import load_data
import tensorflow as tf
import numpy as np

from parser import parse_sub_structure
from tools import extract_axis_1
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
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
Alpha = 0.1
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

    """
    然后是parse里面的batch数据
    """
    parse_input_one_batch_data = []
    parse_input_one_batch_length = []
    parse_input_one_batch_substructures_length = []
    for i in range(batch_size):
        one_sentence = batch_data[i]
        one_sentence_sub_structures = parse_sub_structure(sentence=one_sentence)
        parse_input_one_batch_substructures_length.append(len(one_sentence_sub_structures))
        one_sentence_sub_structures_data = []
        one_sentence_sub_structures_length = []
        for each_sub_structure in one_sentence_sub_structures:
            each_sub_structure_data, each_sub_structure_length = _embedding_one_sentence(each_sub_structure)
            one_sentence_sub_structures_data.append(each_sub_structure_data)
            one_sentence_sub_structures_length.append(each_sub_structure_length)
        parse_input_one_batch_data.extend(one_sentence_sub_structures_data)
        parse_input_one_batch_length.extend(one_sentence_sub_structures_length)
    parse_input_one_batch_data = np.array(parse_input_one_batch_data)
    parse_input_one_batch_length = np.array(parse_input_one_batch_length)
    parse_input_one_batch_substructures_length = np.array(parse_input_one_batch_substructures_length)
    return main_input_one_batch_data, batch_one_labels, main_input_one_batch_length, parse_input_one_batch_data, parse_input_one_batch_length, parse_input_one_batch_substructures_length


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
然后是parse部分RNN模型
"""
with tf.variable_scope('parse'):
    # parse_input_data = tf.placeholder(dtype=tf.int32, shape=(None, max_seq_length))
    # parse_X = tf.nn.embedding_lookup(wordVectors, parse_input_data)
    # parse_sequence_length = tf.placeholder(dtype=tf.int32, shape=(None,))
    # parse_substructure_length = tf.placeholder(dtype=tf.int32, shape=(None,))
    # parse_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    # parse_cell = tf.contrib.rnn.DropoutWrapper(parse_cell, output_keep_prob=0.8)
    # parse_outputs, _ = tf.nn.dynamic_rnn(cell=parse_cell, inputs=parse_X, sequence_length=parse_sequence_length,
    #                                      dtype=tf.float32)
    # parse_h_state = extract_axis_1(parse_outputs, parse_sequence_length - 1)
    # start = 0
    # for i in range(batch_size):
    #     end = start + parse_substructure_length[i]
    #     inner_product = tf.matmul(tf.reshape(main_h_state[i], shape=(1, hidden_size)),
    #                               tf.transpose(parse_h_state[start:end]))
    #     p = tf.nn.softmax(logits=inner_product)
    #     one_sentence_result = tf.reduce_sum(tf.multiply(parse_h_state[start:end], tf.transpose(p)),
    #                                         axis=0, keep_dims=True)
    #     if i == 0:
    #         parse_h_state_new = one_sentence_result
    #     else:
    #         parse_h_state_new = tf.concat([parse_h_state_new, one_sentence_result], 0)
    #     start = end
    Attention_W_1 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], stddev=0.1), dtype=tf.float32)
    Attention_W_2 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], stddev=0.1), dtype=tf.float32)
    Attention_b = tf.Variable(tf.truncated_normal([hidden_size, 1], stddev=0.1), dtype=tf.float32)
    Attention_V = tf.Variable(tf.truncated_normal([1, hidden_size], stddev=0.1), dtype=tf.float32)
    parse_input_data = tf.placeholder(dtype=tf.int32, shape=(None, max_seq_length))
    parse_X = tf.nn.embedding_lookup(wordVectors, parse_input_data)
    parse_sequence_length = tf.placeholder(dtype=tf.int32, shape=(None,))
    parse_substructure_length = tf.placeholder(dtype=tf.int32, shape=(None,))
    parse_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    parse_cell = tf.contrib.rnn.DropoutWrapper(parse_cell, output_keep_prob=0.8)
    parse_outputs, _ = tf.nn.dynamic_rnn(cell=parse_cell, inputs=parse_X, sequence_length=parse_sequence_length,
                                         dtype=tf.float32)
    parse_h_state = extract_axis_1(parse_outputs, parse_sequence_length - 1)
    start = 0
    for i in range(batch_size):
        end = start + parse_substructure_length[i]
        W_1_h_1 = tf.matmul(Attention_W_1, tf.reshape(main_h_state[i], shape=(hidden_size, 1)))
        W_2_h_2 = tf.matmul(Attention_W_2, tf.transpose(parse_h_state[start:end]))
        Attention = tf.matmul(Attention_V, tf.nn.relu(tf.add(tf.add(W_1_h_1, W_2_h_2), Attention_b)))
        p = tf.nn.softmax(logits=Attention)
        one_sentence_result = tf.reduce_sum(tf.multiply(parse_h_state[start:end], tf.transpose(p)),
                                            axis=0, keep_dims=True)
        if i == 0:
            parse_h_state_new = one_sentence_result
        else:
            parse_h_state_new = tf.concat([parse_h_state_new, one_sentence_result], 0)
        start = end
"""
最后是merge部分
"""
with tf.variable_scope('merge'):
    # Define weights
    labels = tf.placeholder(tf.float32, [None, class_num])
    W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
    y_pre = tf.matmul(tf.add((1 - Alpha) * parse_h_state_new, Alpha * main_h_state), W) + bias
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
    logdir = "tensorboard/new_{}_{}/".format(int(time.time()), Alpha)
    modeldir = "models/new_models_{}/".format(Alpha)
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
        main_input_batch_data, batch_labels, main_input_batch_length, parse_input_batch_data, parse_input_batch_length, parse_input_batch_substructures_length = generate_a_batch_data(
            train_data[random_num:], train_labels[random_num:])
        P_valid_tem, _, loss_temp, current_loss_summary = sess.run([p, optimizer, loss, loss_summary], {
            main_input: main_input_batch_data,
            main_sequence_length: main_input_batch_length,
            parse_input_data: parse_input_batch_data,
            parse_sequence_length: parse_input_batch_length,
            parse_substructure_length: parse_input_batch_substructures_length,
            labels: batch_labels
        })
        print(time.strftime("%H:%M:%S"), 'step', i, 'loss', loss_temp)
        if (i != 0 and i % 5 == 0) or i == 1:
            accuracy_temp = 0.0
            current_accuracy_summary = None
            for iter_time in range(20):
                random_num = randint(0, 49)
                random_num = random_num * batch_size
                main_input_batch_data, batch_labels, main_input_batch_length, parse_input_batch_data, parse_input_batch_length, parse_input_batch_substructures_length = generate_a_batch_data(
                    valid_data[random_num:], valid_labels[random_num:])
                one_accuracy_temp, current_accuracy_summary = sess.run([accuracy, accuracy_summary], {
                    main_input: main_input_batch_data,
                    main_sequence_length: main_input_batch_length,
                    parse_input_data: parse_input_batch_data,
                    parse_sequence_length: parse_input_batch_length,
                    parse_substructure_length: parse_input_batch_substructures_length,
                    labels: batch_labels
                })
                accuracy_temp += one_accuracy_temp
            accuracy_temp /= 20
            writer.add_summary(current_loss_summary, global_step=i)
            writer.add_summary(current_accuracy_summary, global_step=i)
            print(time.strftime("%H:%M:%S"), 'step:', i, 'accuracy:', accuracy_temp, 'loss:', loss_temp)
            if accuracy_temp >= max_accuracy:
                save_path = saver.save(sess, "{}{}.ckpt".format(modeldir, accuracy_temp), global_step=i)
                print("saved to %s" % save_path)
                max_accuracy = accuracy_temp
    writer.close()


"""
测试模型
"""


def test_model():
    print('START TEST')
    model_file = tf.train.latest_checkpoint("models/new_models_{}/".format(Alpha))
    saver.restore(sess, model_file)
    accuracy_all = 0.0
    for iter_time in range(150):
        random_num = iter_time * 50
        main_input_batch_data, batch_labels, main_input_batch_length, parse_input_batch_data, parse_input_batch_length, parse_input_batch_substructures_length = generate_a_batch_data(
            test_data[random_num:], test_labels[random_num:])
        accuracy_temp = sess.run(accuracy, {
            main_input: main_input_batch_data,
            main_sequence_length: main_input_batch_length,
            parse_input_data: parse_input_batch_data,
            parse_sequence_length: parse_input_batch_length,
            parse_substructure_length: parse_input_batch_substructures_length,
            labels: batch_labels
        })
        print('iter time {} , accuracy {}'.format(iter_time, accuracy_temp))
        accuracy_all += accuracy_temp
    accuracy_all /= 150
    print('Test Result: {}'.format(accuracy_all))


if __name__ == "__main__":
    train_model()
    test_model()
