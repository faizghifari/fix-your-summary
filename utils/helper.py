import re
import spacy

from utils.labels_tags import label_map

nlp = spacy.load('en_core_web_lg')

def update_labels(labels, label_type, start_idx, end_idx, new_token):
    new_labels = []
    token_length = len(new_token.split())
    for i in range(len(labels)):
        if i < start_idx:
            new_labels.append(labels[i])
        elif i == start_idx:
            new_labels.append(f"B-{label_type}")
            for _ in range(1, token_length):
                new_labels.append(f"I-{label_type}")
        elif i > start_idx and i < end_idx:
            continue
        else:
            new_labels.append(labels[i])

    return new_labels

def add_labels(labels, label_type, end_idx, new_token):
    new_labels = []
    token_length = len(new_token.split())
    for i in range(len(labels)):
        if i < end_idx:
            new_labels.append(labels[i])
        elif i == end_idx:
            new_labels.append("O")
            new_labels.append(f"B-{label_type}")
            for _ in range(1, token_length):
                new_labels.append(f"I-{label_type}")
        else:
            new_labels.append(labels[i])

    return new_labels

def align_distortion_map(distortion_map, end_idx, added_length):
    for dmap in distortion_map:
        if dmap["start_idx"] >= end_idx:
            dmap["start_idx"] += added_length
            dmap["end_idx"] += added_length
    return distortion_map

def extract_digits(num):
    num_ = None
    if num.text.isdigit():
        num_ = int(num.text)
    else:
        num_= int(''.join(filter(str.isdigit, num.text)))
    
    return num_

def extract_digits_str(num):
    num_ = None
    if num.isdigit():
        num_ = int(num)
    else:
        num_= int(''.join(filter(str.isdigit, num)))
    
    return num_

def is_match_case(a, b):
    if a == b:  # check if strings are equal
        return True
    elif a.isupper() and b.isupper():
        return True
    elif a.islower() and b.islower():
        return True
    elif a[0].isupper() and a[1:].islower() and b[0].isupper() and b[1:].islower():
        return True
    else:
        return False

def remove_spaces_and_lower(str1):
    # Remove all types of whitespaces and convert both strings to lowercase
    return re.sub(r'\s', '', str1).lower()

def compare_strings(str1, str2):
    str1 = remove_spaces_and_lower(str1)
    str2 = remove_spaces_and_lower(str2)

    # Compare the modified strings
    return str1 == str2

def tokenize_summary(summary):
    summary_doc = nlp(summary)
    raw_summary = " ".join([token.text for token in summary_doc])

    return raw_summary

def align_tokenized_text(tokenized_tokens):
    aligned_tokens = []
    for i, token in enumerate(tokenized_tokens):
        if i > 0 and token.startswith('##'):
            aligned_tokens[-1] += token[2:]
        else:
            aligned_tokens.append(token)
    
    return aligned_tokens

def map_predictions(raw_preds, word_ids):
    pred_map = {}
    for idx, pred in enumerate(raw_preds):
        word_id = word_ids[idx]
        if word_id in pred_map.keys():
            pred_map[word_id].add(pred)
        else:
            pred_map[word_id] = {pred}

    return pred_map

def align_pred_labels(pred_map, summary_tokens):
    pred_labels = []
    for i, token in enumerate(summary_tokens):
        label = list(pred_map[i])[0]
        pred_labels.append(label)
    
    return pred_labels

def find_missing_numbers(word_ids):
    max_num = max(word_ids)
    complete_list = set(range(max_num + 1))
    num_set = set(word_ids)
    missing_nums = sorted(list(complete_list - num_set))

    return missing_nums

def get_hallucinated_word_list(pred_labels, summary_tokens, label_type, convert_label=True):
    if convert_label:
        id2label = label_map[label_type]["id2label"]
        pred_labels = [id2label[label] for label in pred_labels]
    
    word_list = []
    current_word = []
    prev_labels = "O"
    for token, label in zip(summary_tokens, pred_labels):
        current_label = label[-1]
        if current_label != "O":
            if not current_word:
                current_word.append(token)
            else:
                if prev_labels != current_label:
                    word_list.append(" ".join(current_word))
                    current_word = [token]
                else:
                    current_word.append(token)
        else:
            if current_word:
                word_list.append(" ".join(current_word))
                current_word = []
        prev_labels = current_label

    word_list = "; ".join(word_list) + ";"
    return word_list
