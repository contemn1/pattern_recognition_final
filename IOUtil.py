import pandas
import re
from nltk.tokenize import sent_tokenize
import regex
from functools import reduce
import logging
import sys
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split

REFERENCES = re.compile("REFERENCES|References|Rererences")
ESCAPE = re.compile("\n+")
EQUATION = re.compile("\b?:?=.*(\.|!|,|d[a-z]|\))$")
EMAIL = re.compile("\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}")
FLOAT_NUMBER = re.compile("[~0-9\s]\.[0-9]")
CONTINOUS_PUNCT = re.compile("[,\.~+-;\s0-9'\"]{2,}")
BRACKET = re.compile("[{(\[][A-Za-z0-9_\s,]+[})\]]")
STRANGE_CHARACTER = regex.compile("[^\w\p{P}\s\+\*=\-\>\<]+")
CONTAINS_BRACKET = regex.compile("\{.*\}|\[.*\]|\(.*\)")


def read_file(file_path, preprocess):
    try:
        with open(file_path, encoding="utf8") as file:
            new_results = []
            contents = file.readlines()
            for sentence in contents:
                result = preprocess(sentence)
                if result:
                    new_results.append(result)

            return new_results
    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def output_file(file_path, sent_list):
    try:
        with open(file_path, mode="w+") as f:
            for line in sent_list:
                f.write(line)

    except IOError as err:
        print("Failed_to_open_file {0}".format(err))


def split_string(input):
    return input.split(" ", 1)


def read_csv(file_path):
    a = pandas.read_csv(file_path)

    b = a["paper_text"]
    b = [REFERENCES.split(ele, 1)[0] for ele in b]
    first = regex.compile("[^\P{P}\.,\(\)'\?]{2,}")
    third = regex.compile("[~]+|[\.\?]{2,}")
    escape = regex.compile("\s{2,}")
    pure_punct = regex.compile("^[\p{P}\s]+$")

    regex_list = [first, CONTAINS_BRACKET, third, pure_punct, escape]
    for ele in b:
        arr = ele.split("\n")
        arr = [reduce(lambda first, second: second.sub(" ", first), regex_list, line) for line in arr if line]
        arr = [ele.strip() for ele in arr]
        arr = [ele for ele in arr if len(ele.split(" ")) >= 3]
        new_str = "\n".join(arr)
        new_arr = sent_tokenize(new_str)
        for ele in new_arr:
            line = ESCAPE.sub(" ", ele)
            if len(line.split(" ")) >= 4 and not STRANGE_CHARACTER.search(line):
                print(line.strip())


def filter_raw_csv(file_path):
    sents = read_file(file_path, preprocess=lambda x: CONTAINS_BRACKET.sub("", x))
    res = [ele for ele in sents if not re.search("=|>|<", ele)]
    for ele in res:
        line = ele.strip()
        if len(line.split(" ")) > 4:
            print(line)


def get_word_dict(sentences, tokenize=True):
    # create vocab of words
    word_dict = {}
    sentences = [s.split() if not tokenize else word_tokenize(s)
                 for s in sentences]
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    return word_dict


def get_glove(glove_path, word_dict):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.fromstring(vec.strip(), sep=' ')

    print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))

    return word_vec


def get_glove_nips(glove_path):
    word_vec = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_vec[word] = np.fromstring(vec.strip(), sep=' ')

    return word_vec


def tokenizer(use_nitk):
    def tokenize(sentences):
        if use_nitk:
            return word_tokenize(sentences)
        else:
            return sentences.split(" ")
    return tokenize

def remove_words_not_in_glove(sentence, glove_dict):
    words = word_tokenize(text=sentence)
    words = [word for word in words if word in glove_dict]
    return words


if __name__ == '__main__':
    DATA_PATH = "/Users/zxj/Downloads/nips_data/training_data/"
    file_paths = ["nips_train.txt", "nips_valid.txt", "nips_test.txt"]
    out_paths = ["nips_train_sorted.txt", "nips_valid_sorted.txt", "nips_test_sorted.txt"]

    file_paths = [DATA_PATH + path for path in file_paths]
    files = [read_file(path, lambda x: x) for path in file_paths]
    for index in range(len(files)):
        ele = files[index]
        ele = [sent.split(" ") for sent in ele]
        ele = [arr for arr in ele if 45 >= len(arr) >= 7]
        ele.sort(key=lambda x: -len(x))
        ele = [" ".join(arr) for arr in ele]
        print(len(ele))
        output_file(out_paths[index], ele)