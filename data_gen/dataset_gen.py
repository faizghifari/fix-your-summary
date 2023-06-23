import random
import argparse

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

from data_gen.entities import replace_entities
from data_gen.quantity import replace_quantity
from data_gen.pronoun import replace_pronoun
from data_gen.noun import replace_noun
from data_gen.verb import replace_verb

from data_gen.json_data_gen import generate_json_data

from utils.helper import nlp, tokenize_summary
from utils.labels_tags import label_map, convert_labels_remove_bio, convert_labels_to_one_type

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
label2id = label_map["labels_sep_bio"]["label2id"]

# current parameters use for corrupted dataset generation, modify this to try different config
distort_funcs = [
    {"func": replace_entities, "Y": 1, "alpha": 0.5},
    {"func": replace_pronoun, "Y": 1, "alpha": 0.5},
    {"func": replace_quantity, "Y": 1, "alpha": 0.5},
    {"func": replace_noun, "Y": 0.3, "alpha": 0.3},
    {"func": replace_verb, "Y": 0.3, "alpha": 0.3},
]

def distort_data(summary, labels, distortion_map, dialogue, split):
    for func_info in distort_funcs:
        func = func_info["func"]
        Y = func_info["Y"]
        alpha = func_info["alpha"]

        assert 0 < Y <= 1

        rand_num = random.random()
        if rand_num <= Y:
            summary = " ".join(summary.split())
            summary, labels, distortion_map = func(summary, labels, distortion_map, alpha=alpha, dialog=dialogue, split=split)

    return summary, labels, distortion_map

def create_data_point(data, distort, split):
    summ = " ".join(data["summary"].split())
    raw_summary = tokenize_summary(summ)
    while summ != raw_summary:
        summ = raw_summary
        raw_summary = tokenize_summary(summ)

    raw_labels = ["O"] * len(raw_summary.split())
    raw_distortion_map = []
    summary = raw_summary
    labels = raw_labels
    distortion_map = raw_distortion_map[:]
    if distort:
        summary, labels, distortion_map = distort_data(raw_summary, raw_labels, raw_distortion_map, data["dialogue"], split)
    summary_doc = nlp(summary)
    summary = summary.split()
    
    while len(summary) != len(labels) or len(summary) != len(summary_doc):
        summary, labels, distortion_map = distort_data(raw_summary, raw_labels, raw_distortion_map, data["dialogue"], split)
        summary_doc = nlp(summary)
        summary = summary.split()
    
    return summary, labels, distortion_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corrupted_data_ratio", type=int, required=True,
                    help="Corrupted data ratio")

    parser.add_argument("--max_len", type=int, default=512,
                        help="Maximum length")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    
    args = parser.parse_args()

    dataset = load_from_disk(args.data_dir)
    
    X = args.corrupted_data_ratio
    max_len = args.max_len

    new_dataset = DatasetDict()
    for split in ["validation", "test", "train"]:
        ids = []
        dialogues = []
        ref_summaries = []
        distorted_summaries = []
        distortion_maps = []
        labels_sep_bio = []
        tag_labels_sep_bio = []
        labels_sep = []
        tag_labels_sep = []
        labels_bio = []
        tag_labels_bio = []
        labels = []
        tag_labels = []

        skipped_data = 0

        total_data = len(dataset[split])
        num_to_distort = int(X * total_data)
        indices_to_distort = random.sample(range(total_data), num_to_distort)

        for i, data in tqdm(enumerate(dataset[split])):
            try:
                distorted_summary, raw_labels, distortion_map = create_data_point(data, i in indices_to_distort, split)
            except Exception as e:
                print(i)
                raise Exception()
            
            distortion_map = sorted(distortion_map, key=lambda d: d['start_idx'])
            dialog_doc = nlp(data["dialogue"])
            dialog = [token.text for token in dialog_doc]

            if (raw_labels is None) or (len(tokenizer(dialog, distorted_summary, is_split_into_words=True).tokens()) > max_len):
                skipped_data += 1
                continue
            
            label_sep_bio = [label2id[l] for l in raw_labels]
            tag_label_sep_bio = raw_labels
            tag_label_sep = convert_labels_remove_bio(tag_label_sep_bio)
            tag_label_bio = convert_labels_to_one_type(tag_label_sep_bio)
            tag_label = convert_labels_remove_bio(tag_label_bio)
            label_sep = [label_map["labels_sep"]["label2id"][l] for l in tag_label_sep]
            label_bio = [label_map["labels_bio"]["label2id"][l] for l in tag_label_bio]
            label = [label_map["labels"]["label2id"][l] for l in tag_label]

            ids.append(data["id"])
            dialogues.append(dialog)
            ref_summaries.append(" ".join(data["summary"].split()))
            distorted_summaries.append(distorted_summary)
            distortion_maps.append(distortion_map)
            labels_sep_bio.append(label_sep_bio)
            tag_labels_sep_bio.append(tag_label_sep_bio)
            labels_sep.append(label_sep)
            tag_labels_sep.append(tag_label_sep)
            labels_bio.append(label_bio)
            tag_labels_bio.append(tag_label_bio)
            labels.append(label)
            tag_labels.append(tag_label)

        new_dataset[split] = Dataset.from_dict({
            "ids": ids,
            "dialogues": dialogues,
            "ref_summaries": ref_summaries,
            "distorted_summaries": distorted_summaries,
            "distortion_maps": distortion_maps,
            "labels_sep_bio": labels_sep_bio,
            "tag_labels_sep_bio": tag_labels_sep_bio,
            "labels_sep": labels_sep,
            "tag_labels_sep": tag_labels_sep,
            "labels_bio": labels_bio,
            "tag_labels_bio": tag_labels_bio,
            "labels": labels,
            "tag_labels": tag_labels,
        })
        print(f"Skipped data for {split}: {skipped_data}")

    new_dataset.save_to_disk(args.output_dir)

    generate_json_data(args.output_dir, new_dataset)
