import tensorflow as tf
import numpy as np
import time
import re
import operator
from pprint import pprint
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib import seq2seq
from tensorflow.contrib import layers

lines = open('movie_lines.txt', encoding="utf-8", errors="ignore").read().split("\n")
conversations = open('movie_conversations.txt', encoding="utf-8", errors="ignore").read().split("\n")

id2line = dict()

for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[-1]

conversations_list = list()

for conversation in conversations:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_list.append(_conversation.split(","))

questions = list()
answers = list()

for conversation in conversations_list:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"won't", "would not", text)
    text = re.sub(r"was't", "was not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"[!.@#$%^&*()_<>?/+-/*|`~{};:,]", "", text)
    return text


clean_qusetions = list()
clean_answers = list()

for question in questions:
    clean_qusetions.append(clean_text(question))

for answer in answers:
    clean_answers.append(clean_text(answer))

word_count = dict()

for question in clean_qusetions:
    for word in question.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1

preprocess_questions = dict()
preprocess_answers = dict()
threshold = 20
word_count_increment = 0

for key, value in word_count.items():
    if value >= threshold:
        preprocess_questions[key] = word_count_increment
        word_count_increment += 1

word_count_increment = 0
for key, value in word_count.items():
    if value >= threshold:
        preprocess_answers[key] = word_count_increment
        word_count_increment += 1

tokens = ['<PAD>', '<EOS>', '<SOS>', '<OUT>']

for token in tokens:
    preprocess_questions[token] = len(preprocess_questions) + 1

for token in tokens:
    preprocess_answers[token] = len(preprocess_answers) + 1

answerint2word = dict()

answerint2word = {value: key for key, value in preprocess_answers.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

question_to_int = []
for question in clean_qusetions:
    ints = []
    for word in question.split():
        if word not in preprocess_questions:
            ints.append(preprocess_questions['<OUT>'])
        else:
            ints.append(preprocess_questions[word])
    question_to_int.append(ints)

answer_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in preprocess_answers:
            ints.append(preprocess_answers['<OUT>'])
        else:
            ints.append(preprocess_answers[word])
    answer_to_int.append(ints)

sorted_questions = []
sorted_answers = []

for length in range(1, 25 + 1):
    for i in enumerate(question_to_int):
        if len(i[1]) == length:
            sorted_questions.append(question_to_int[i[0]])
            sorted_answers.append(answer_to_int[i[0]])


def rnn_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    lr = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return inputs, targets, lr, keep_prob


def rnn_training_data(batch_size, targets, word2int):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, begin=[0, 0], end=[batch_size, -1], strides=[1, 1])
    preproccessed_targets = tf.concat([left_side, right_side], axis=1)
    return preproccessed_targets


def rnn_encoder(rnn_inputs, rnn_size, keep_prob, num_of_layers, sequence_length):
    lstm = BasicLSTMCell(input_size=rnn_size)
    lstm_dropout = DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = MultiRNNCell([lstm_dropout] * num_of_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, cell_bw=encoder_cell,
                                                       sequence_length=sequence_length, dtype=tf.float32,
                                                       inputs=rnn_inputs)
    return encoder_state


def decoder_train_set(encoder_state, decoder_cell, batch_size, decoder_scope, keep_prob, decoder_embedded_input,
                      sequence_length, decoder_output_function):
    attention_state = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_key, attention_value, attention_score_function, attention_construct_function = seq2seq.prepare_attention(
        attention_states=attention_state, attention_option="nabdanau",
        num_units=decoder_cell.output_size)
    decoder_train_output, _, _ = seq2seq.attention_decoder_fn_train(encoder_state=encoder_state[0],
                                                                    attention_keys=attention_key,
                                                                    attention_values=attention_value,
                                                                    attention_score_fn=attention_score_function,
                                                                    attention_construct_fn=attention_construct_function,
                                                                    name="attn_dec_train")
    decoder_output = seq2seq.dynamic_rnn_decoder(decoder_cell, decoder_train_output,
                                                 inputs=decoder_embedded_input,
                                                 sequence_length=sequence_length, scope=decoder_scope)

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob=keep_prob)

    return decoder_output_function(decoder_output_dropout)


def decoder_test_set(encoder_state, decoder_cell, batch_size, decoder_scope, keep_prob, decoder_embedding_matrix,
                     sequence_length, decoder_output_function, sos_id, eos_id, max_length, num_symbols):
    attention_state = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_key, attention_value, attention_score_function, attention_construct_function = seq2seq.prepare_attention(
        attention_states=attention_state, attention_option="nabdanau",
        num_units=decoder_cell.output_size)
    decoder_test_output = seq2seq.attention_decoder_fn_inference(output_fn=decoder_output_function,
                                                                 encoder_state=encoder_state,
                                                                 attention_keys=attention_key,
                                                                 attention_values=attention_value,
                                                                 attention_score_fn=attention_score_function,
                                                                 attention_construct_fn=attention_construct_function,
                                                                 embeddings=decoder_embedding_matrix,
                                                                 start_of_sequence_id=sos_id,
                                                                 end_of_sequence_id=eos_id,
                                                                 maximum_length=max_length,
                                                                 num_decoder_symbols=num_symbols,
                                                                 dtype=tf.float32,
                                                                 name="attn_dec_inf"
                                                                 )
    decoder_output, _, _ = seq2seq.dynamic_rnn_decoder(decoder_cell, decoder_test_output,
                                                       scope=decoder_scope)

    return decoder_output


def decoder_rnn(rnn_size, keep_prob, num_of_layers, num_of_words, encoder_state, word2int, batch_size,
                decoder_embedded_input, decoder_embedding_matrix, sequence_length):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = BasicLSTMCell(rnn_size)
        lstm_dropout = DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = MultiRNNCell([lstm_dropout] * num_of_layers)
        weight = tf.truncated_normal_initializer(stddev=0.1)
        biase = tf.zeros_initializer()

        output_function = lambda x: layers.fully_connected(x, num_outputs=num_of_words, activation_fn=None,
                                                           normalizer_fn=None, weights_initializer=weight,
                                                           biases_initializer=biase,
                                                           scope=decoding_scope)

        decoder_trainings = decoder_train_set(encoder_state, decoder_cell, batch_size, decoding_scope, keep_prob,
                                              decoder_embedded_input,
                                              sequence_length, output_function)

        decoding_scope.reuse_variables()

        decoder_predictions = decoder_test_set(encoder_state, decoder_cell, batch_size, decoding_scope, keep_prob,
                                               decoder_embedding_matrix,
                                               sequence_length, output_function, word2int['<SOS>'], word2int['<EOS>'],
                                               sequence_length - 1,
                                               num_of_words)

        return decoder_trainings, decoder_predictions


def seq2seq(inputs, targets, batch_size, questionword2int, encoder_embedded_size, decoder_embedding_size,
            questions_num_words, answer_num_word, rnn_size, keep_prob, num_of_layers, sequence_length):
    encoder_embedded_input = layers.embed_sequence(inputs, encoder_embedded_size, answer_num_word,
                                                   initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = rnn_encoder(encoder_embedded_input, rnn_size, keep_prob, num_of_layers, sequence_length)
    preprocessing_targets = rnn_training_data(batch_size, targets, questionword2int)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessing_targets)

    training_prediction, test_predictions = decoder_rnn(rnn_size, keep_prob, num_of_layers, questions_num_words,
                                                        encoder_state, questionword2int, batch_size,
                                                        decoder_embedded_input, decoder_embeddings_matrix,
                                                        sequence_length)

    return training_prediction, test_predictions
