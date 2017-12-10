import pandas
import re
from nltk.tokenize import sent_tokenize
import regex
from functools import reduce
import logging
import sys

REFERENCES = re.compile("REFERENCES|References|Rererences")
ESCAPE = re.compile("\n+")
EQUATION = re.compile("\b?:?=.*(\.|!|,|d[a-z]|\))$")
EMAIL = re.compile("\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}")
FLOAT_NUMBER = re.compile("[~0-9\s]\.[0-9]")
CONTINOUS_PUNCT = re.compile("[,\.~+-;\s0-9'\"]{2,}")
BRACKET = re.compile("[{(\[][A-Za-z0-9_\s,]+[})\]]")
THINKREGEX = re.compile(" says? | said | knows? | knew | thinks? | thought ")



def read_file(file_path, preprocess):
    content_list = []
    try:
        with open(file_path, encoding="utf8") as file:
            contents = file.readlines()
            for sentence in contents:
                content_list.append(preprocess(sentence))

            return content_list
    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def split_string(input):
    return input.split(" ", 1)


def read_csv(file_path):
    a = pandas.read_csv(file_path)
    b = a["paper_text"]
    content_no_references = [REFERENCES.split(ele, 1)[0] for ele in b]
    for no_reference in content_no_references:
        content_no_email = EMAIL.sub("", no_reference)
        for ele in sent_tokenize(content_no_email):
            no_float = FLOAT_NUMBER.sub(" ", ele)
            no_equation = EQUATION.sub(" ", no_float)
            no_bracket = BRACKET.sub(" ", no_equation)
            no_rebundant_punct = CONTINOUS_PUNCT.sub(" ", no_bracket)
            no_escape = ESCAPE.sub(" ", no_rebundant_punct)
            no_rebundant_space = re.sub("\s{2,}", " ", no_escape)
            if no_rebundant_space:
                words = no_rebundant_space.split(" ")
                words = [word.lower() for word in words if len(words) >= 3]
                if words:
                    print(" ".join(words))


def read_csv2(file_path):
    a = pandas.read_csv(file_path)

    b = a["paper_text"]
    b = [REFERENCES.split(ele, 1)[0] for ele in b]
    first = regex.compile("[^\P{P}\.,\(\)'\?]{2,}")
    second = regex.compile("\(.+\)")
    third = regex.compile("[~]+|[\.\?]{2,}")
    escape = regex.compile("\s{2,}")
    pure_punct = regex.compile("^[\p{P}\s]+$")

    regex_list = [first, second, third, pure_punct]
    for ele in b:
        arr = ele.split("\n")
        arr = [reduce(lambda first, second: second.sub("", first), regex_list, line) for line in arr if line]
        arr = [ele.strip() for ele in arr]
        arr = [ele for ele in arr if len(ele.split(" ")) >= 3]
        new_str = "\n".join(arr)
        new_arr = sent_tokenize(new_str)
        for ele in new_arr:
            line = ESCAPE.sub("", ele)
            if len(line.split(" ")) >= 4:
                print(line.strip())


def get_word_dict(sentences, tokenize=True):
    # create vocab of words
    word_dict = {}
    if tokenize:
        from nltk.tokenize import word_tokenize
    sentences = [s.split() if not tokenize else word_tokenize(s)
                 for s in sentences]
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word.lower()] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    return word_dict


def get_glove(glove_path, word_dict):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word.lower() in word_dict:
                word_vec[word.lower()] = vec

    print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))

    return word_vec


def read_text_file(input_path):
    sentecnes = []
    try:
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                results = sent_tokenize(line)
                sents = [ele1 for ele1 in results if THINKREGEX.search(ele1)]
                for ele in sents:
                    if len(ele.split(" ")) < 30:
                        print(ele)

            return sentecnes
    except IOError as err:
        print("Failed to read file {0}".format(err))
        return sentecnes


if __name__ == '__main__':
    read_csv2("/Users/zxj/Downloads/nips-papers/papers.csv")