import pandas
import re
from nltk.tokenize import sent_tokenize


REFERENCES = re.compile("REFERENCES|References")
ESCAPE = re.compile("\n+")
EQUATION = re.compile("\s?:?=.*(\.|!|,|d[a-z]|\))")
EMAIL = re.compile("\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}")
FLOAT_NUMBER = re.compile("[~0-9\s]\.[0-9]")
CONTINOUS_PUNCT = re.compile("[,\.~+-;iI\s0-9'\"]{2,}")
BRACKET = re.compile("[{(][A-Za-z0-9_\s,]+[})]")


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
            no_rebundant_space = re.sub("\s+", " ", no_escape)
            if no_rebundant_space:
                words = no_rebundant_space.split(" ")
                words = [word.lower() for word in words if len(words) >= 3]
                if words:
                    print(" ".join(words))


def deal_with_api(input_string):
    original_card = re.compile("card=[0-9]{15,16}")
    cards = original_card.search(input_string)
    token = "XXXXXXXXXXXXXXXXXXXX"
    if cards:
        cards_string = cards.group()
        result = cards_string[:11] + token + cards_string[-4:]
        output = original_card.sub(result, input_string)
        return result, cards_string, output

    return ()


def deal_with_bank(input_string, target_dict):
    card_with_crytophy = re.compile("card=[0-9]{6}X{20}[0-9]{4}")
    cards_part = card_with_crytophy.search(input_string)
    if cards_part:
        cards_part = cards_part.group()
        if cards_part in target_dict:
            return card_with_crytophy.sub(target_dict[cards_part], input_string)

    else:
        return input_string


def deal_with_inputs(lines):
    cards_dict = {}
    result_list = []
    for input_string in lines:
        if input_string.startswith("API"):
            res_tuple = deal_with_api(input_string)
            if res_tuple:
                result, cards_string, output_string = deal_with_api(input_string)
                cards_dict[result] = cards_string
                result_list.append(output_string)
            else:
                result_list.append(input_string)

        if input_string.startswith("BANK"):
            result_list.append(deal_with_bank(input_string, cards_dict))

    return result_list


if __name__ == '__main__':
    input1 = ["API:amount=301&card=5656565656565656",
              "BANK:card=565656XXXXXXXXXXXXXXXXXXXX5656&authorize=true"]

    print(deal_with_inputs(input1))
