import tensorflow as tf
import numpy as np
import time
import re
import operator
from pprint import pprint

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

answerint2word = {value:key for key, value in preprocess_answers.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

question_to_int=[]
for question in clean_qusetions:
    ints=[]
    for word in question.split():
        if word not in preprocess_questions:
            ints.append(preprocess_questions['<OUT>'])
        else:
            ints.append(preprocess_questions[word])
    question_to_int.append(ints)

answer_to_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in preprocess_answers:
            ints.append(preprocess_answers['<OUT>'])
        else:
            ints.append(preprocess_answers[word])
    answer_to_int.append(ints)

sorted_questions=[]
sorted_answers=[]

for length in range(1,25+1):
    for i in enumerate(question_to_int):
        if len(i[1]) == length:
            sorted_questions.append(question_to_int[i[0]])
            sorted_answers.append(answer_to_int[i[0]])


