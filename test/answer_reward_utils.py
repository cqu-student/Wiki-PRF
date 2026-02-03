import subprocess
from typing import TYPE_CHECKING, Dict, Union
import functools
import numpy as np
import re
import string
import os
from transformers import BertTokenizer
from torch.nn import functional as F
import torch
from eval_utils import evaluate_example

#################################################
"""InfoSeek Evaluation Script."""

import re
import json
import string
from typing import Any, Dict, Generator, List, Tuple, Union


def normalize_answer(text: str) -> str:
    """Normalize a given text by removing articles, punctuation, and white spaces, and converting to lowercase."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lowercase(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lowercase(text))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if the normalized prediction exactly matches the normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(
    metric_fn,
    prediction: str,
    ground_truths: List[str]
    ) -> Union[int, bool]:
    """Compute the maximum score of a prediction over a list of ground truths using a given metric function."""
    return max(
        metric_fn(prediction, ground_truth) for ground_truth in ground_truths
    )


def in_range(number: float, range_list: Tuple[float, float]) -> bool:
    """Check if a number is within the specified range (inclusive)."""
    min_num, max_num = range_list
    return min_num <= number <= max_num


def safe_division(x: float, y: float) -> float:
    """Divide x by y, returning 0 if y is 0."""
    return x / y if y != 0 else 0


def metric_numerical_range(
    pred: Union[float, Tuple[float, float], List[float]],
    answer: Union[float, Tuple[float, float], List[float]],
    tolerance: float = 0.1,
    ) -> int:
    """Scores numerical questions based on ranges and tolerances.

    1) First, convert single number answer to a range with +/- tolerance.
    2) If prediction is a single number, return 1 if it's in the answer range, 0
    otherwise.
    3) If prediction is a range, return 1 if the range is in the answer range or
    if the IOU
        (overlap between prediction and answer range) > 0.5, 0 otherwise.

    Args:
        pred: A list/tuple of 2 numbers or a single number.
        answer: A list/tuple of 2 numbers or a single number.
        tolerance: A float value for the tolerance range (default: 0.1).

    Returns:
        int: 1 if conditions are met, 0 otherwise.
    """
    answer = list(answer) if isinstance(answer, tuple) else answer
    pred = list(pred) if isinstance(pred, tuple) else pred
    if not isinstance(answer, list):
        answer = [answer * (1 - tolerance), answer * (1 + tolerance)]

    # Prediction is a single number
    if not isinstance(pred, list):
        return 1 if in_range(pred, answer) else 0

    # Prediction is a range
    if answer[0] <= pred[0] <= answer[1] and answer[0] <= pred[1] <= answer[1]:
        return 1
    else:
        iou = range_intersection_over_union(pred, answer)
        return 1 if iou >= 0.5 - 1e-12 else 0


def process_numerical_answer(string_number: str) -> Union[float, List[float]]:
    """Parses numerical answer string into numbers (a single number or a range).

    1) Clean the string and extract numbers;
    2) if there are 2 numbers, return a range as [minimum value, maximum value]
        else if there is 1 number, return a single number
        else return [0, 0]

    Args:
        string_number: A string representing a numerical answer.

    Returns:
        A single digit or a list with 2 numbers.
    """
    # Clean string
    string_number = clean_str_range(string_number)
    numerical_numbers_tmp = re.findall(
        r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', string_number
    )
    numerical_numbers_tmp = [
        n.replace(',', '').strip('.') for n in numerical_numbers_tmp
    ]
    numerical_numbers = []
    for n in numerical_numbers_tmp:
        if n.count('.') > 1:
            n = n.split('.')[0]
            numerical_numbers.append(float(n))
        else:
            numerical_numbers.append(float(n))

    # Use the first 2 numbers
    if len(numerical_numbers) > 2:
        numerical_numbers = numerical_numbers[:2]

    if len(numerical_numbers) == 2:
        first_val = numerical_numbers[0]
        second_val = numerical_numbers[1]
        return [first_val, second_val] if first_val <= second_val else first_val
    elif len(numerical_numbers) == 1:
        return numerical_numbers[0]
    else:
        return [0, 0]


def find_all(s: str, c: str) -> Generator[int, None, None]:
    """Find all occurrences of a character in a string and return their indices.

    Args:
        s: The input string to search.
        c: The character to search for.

    Yields:
        int: The index of the next occurrence of the character.
    """
    idx = s.find(c)
    while idx != -1:
        yield idx
        idx = s.find(c, idx + 1)


def clean_str_range(text: str) -> str:
    """Clean range expression in a string (e.g., '9-10' --> '9 - 10').

    Args:
        text: The input string containing the range expression.

    Returns:
        str: The cleaned string with proper spacing around the hyphen.
    """
    idx_list = list(find_all(text, '-'))
    idx_replace = [
        idx for idx in idx_list if idx >= 1 and text[idx - 1].isdigit()
    ]
    new_str = ''.join(
        ' - ' if idx in idx_replace else s for idx, s in enumerate(text)
    )
    return new_str


def range_intersection_over_union(
        x_list: List[float], y_list: List[float]
    ) -> float:
    """Calculate the intersection over union (IOU) of two ranges."""
    min_1, max_1 = min(x_list), max(x_list)
    min_2, max_2 = min(y_list), max(y_list)

    overlap = max(0.0, min(max_1, max_2) - max(min_1, min_2))
    length_x = (max_1 - min_1) + 1e-12
    length_y = (max_2 - min_2) + 1e-12
    iou = safe_division(overlap, length_x + length_y - overlap)
    return iou

##################################################


# _VOCAB_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt'
_VOCAB_PATH = '/bert-base-uncased'
# _MODEL_PATH = 'https://www.kaggle.com/models/google/bert/TensorFlow2/answer-equivalence-bem/1'
# 使用 resolve 获取模型的实际下载地址
# resolved_path = hub.resolve(_MODEL_PATH)

# print(f"模型的实际下载地址: {resolved_path}")
# https://www.kaggle.com/models/google/bert/TensorFlow2/answer-equivalence-bem/1
_MODEL_PATH = '/bem/1'
_PUNCTUATION_CHARACTERS = string.punctuation + '‘’´`_'
_DIGIT_MAP = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
    'entailment': 'yes',
    'true': 'yes',
    'contradiction': 'no',
    'false': 'no',
}
_CONTRACTIONS = {
    'aint': "ain't",
    'arent': "aren't",
    'cant': "can't",
    'couldve': "could've",
    'couldnt': "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    'didnt': "didn't",
    'doesnt': "doesn't",
    'dont': "don't",
    'hadnt': "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    'hasnt': "hasn't",
    'havent': "haven't",
    'hed': "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    'hes': "he's",
    'howd': "how'd",
    'howll': "how'll",
    'hows': "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    'Im': "I'm",
    'Ive': "I've",
    'isnt': "isn't",
    'itd': "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    'itll': "it'll",
    "let's": "let's",
    'maam': "ma'am",
    'mightnt': "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    'mightve': "might've",
    'mustnt': "mustn't",
    'mustve': "must've",
    'neednt': "needn't",
    'notve': "not've",
    'oclock': "o'clock",
    'oughtnt': "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    'shant': "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    'shouldve': "should've",
    'shouldnt': "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": 'somebodyd',
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    'somebodyll': "somebody'll",
    'somebodys': "somebody's",
    'someoned': "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    'someonell': "someone'll",
    'someones': "someone's",
    'somethingd': "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    'somethingll': "something'll",
    'thats': "that's",
    'thered': "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    'therere': "there're",
    'theres': "there's",
    'theyd': "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    'theyll': "they'll",
    'theyre': "they're",
    'theyve': "they've",
    'twas': "'twas",
    'wasnt': "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    'weve': "we've",
    'werent': "weren't",
    'whatll': "what'll",
    'whatre': "what're",
    'whats': "what's",
    'whatve': "what've",
    'whens': "when's",
    'whered': "where'd",
    'wheres': "where's",
    'whereve': "where've",
    'whod': "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    'wholl': "who'll",
    'whos': "who's",
    'whove': "who've",
    'whyll': "why'll",
    'whyre': "why're",
    'whys': "why's",
    'wont': "won't",
    'wouldve': "would've",
    'wouldnt': "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    'yall': "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    'youd': "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    'youll': "you'll",
    'youre': "you're",
    'youve': "you've"
}


def preprocess_answer(
    answer,
    punctuation_characters=_PUNCTUATION_CHARACTERS,
    replacement_character='',
):
    """Function to preprocess VQA answers."""

    def remove_articles(s):
        """Remove common articles and prefixes in the answer."""
        return re.sub(r'\b(the answer is|a|an|the)\b', ' ', s)

    def replace_punctuation(s):
        """Replace punctuation characters."""
        to_replace = set(punctuation_characters)
        return ''.join(replacement_character if c in to_replace else c for c in s)

    def white_space_fix(s):
        """Remove superfluous whitespace."""
        return ' '.join(s.split())

    def remove_llm_span_prefix(answer, prefix='<extra_id_0> '):
        """Remove span prefix added by some LLM."""
        if answer.startswith(prefix):
            return answer.replace(prefix, replacement_character)
        return answer

    def standarize_digits_and_contractions(s):
        """Standarize the representation of some digits and common contractions."""
        output = []
        tmp = s.split()
        for w in tmp:
            w = _DIGIT_MAP.get(w, w)
            w = _CONTRACTIONS.get(w, w)
            output.append(w)
        return ' '.join(output)

    answer = answer.lower().replace('\n', ' ').replace('\t', ' ').strip()
    answer = remove_llm_span_prefix(answer)
    answer = replace_punctuation(answer)
    answer = remove_articles(answer)
    answer = standarize_digits_and_contractions(answer)
    answer = white_space_fix(answer)

    return answer

def standarize_digits_and_contractions(s):
  """Standarize the representation of some digits and common contractions."""
  output = []
  tmp = s.split()
  for w in tmp:
    w = _DIGIT_MAP.get(w, w)
    w = _CONTRACTIONS.get(w, w)
    output.append(w)
  return ' '.join(output)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_scoring_function(reference, candidate, question):
    """Compute exact match between single reference and candidate answers."""
    if isinstance(reference, str):
        preprocessed_reference = preprocess_answer(reference)
        preprocessed_reference = [normalize_answer(preprocessed_reference)]
    elif isinstance(reference, list):
        preprocessed_reference = [preprocess_answer(ref) for ref in reference]
        preprocessed_reference = [normalize_answer(ref) for ref in preprocessed_reference]
    elif isinstance(reference, dict):
        #import pdb; pdb.set_trace()
        candidate = candidate.lower().replace('\n', ' ').replace('\t', ' ').strip()
        preprocessed_candidate = standarize_digits_and_contractions(candidate)
        preprocessed_candidate = process_numerical_answer(preprocessed_candidate)
        score = 0.0
        if preprocessed_candidate != [0,0]:
          score = metric_numerical_range(preprocessed_candidate, reference["wikidata"])
        return score
        """
        我的输出是 ：

'This bird can produce between 6-8 offspring at the same time by laying 6-8 eggs.'

其对应的参考答案为：{'wikidata': 6.9, 'range': [6.210000000000001, 7.590000000000001]}
        """
    else:
        import pdb; pdb.set_trace()
        raise ValueError('Reference must be a string or list of strings.')
    preprocessed_candidate = preprocess_answer(candidate)
    preprocessed_candidate = normalize_answer(preprocessed_candidate)
    for ref in preprocessed_reference:
        if ref in preprocessed_candidate:
            return True
            #return 1.0
    return False
    #score = evaluate_example(question, preprocessed_reference, preprocessed_candidate, question_type="templeted")
    #return score



def vqa_match_function(example):
    
    if not example['reference']:
        raise ValueError("Reference list is empty")
    
    #matches_exactly = exact_match_scoring_function(example['reference'], example['candidate'], example['question'])
    #return matches_exactly
    matches_exactly = exact_match_scoring_function(example['reference'], example['candidate'], example['question'])
    if isinstance(matches_exactly,float):
        return matches_exactly
    elif matches_exactly:
        return 1.0
    return 0



def evaluate_example(candidate, reference_list, question):
    if not reference_list:
        raise ValueError("Reference list is empty")

    scores = []
    for reference in reference_list:
        example = {
            'question': question,
            'reference': reference,
            'candidate': candidate,
        }
        score = vqa_match_function(example)
        scores.append(score)
    return max(scores)



