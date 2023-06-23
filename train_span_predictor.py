import os
import sys
import torch
import logging
import argparse
import evaluate
import numpy as np

from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer

from utils.labels_tags import label_map

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

metric = evaluate.load("seqeval")

def align_labels_with_tokens(labels, word_ids, context_len):
    new_labels = []
    current_word = None
    for i in range(len(word_ids)):
        if i < context_len + 2:
            new_labels.append(-100)
        else:
            if word_ids[i] != current_word:
                # Start of a new word!
                current_word = word_ids[i]
                label = -100 if word_ids[i] is None else labels[word_ids[i]]
                new_labels.append(label)
            else:
                # Special token or same word as prev. token
                new_labels.append(-100)

    return new_labels

def preprocess_data(data, tokenizer, label_key):
    tokenized_inputs = tokenizer(data['dialogues'], data['distorted_summaries'], is_split_into_words=True, truncation=True, max_length=512)
    new_labels = []
    for i, labels in enumerate(data[label_key]):
        word_ids = tokenized_inputs.word_ids(i)
        dialogue = data["dialogues"][i]
        context_len = len(tokenizer.tokenize(dialogue, is_split_into_words=True))
        new_labels.append(align_labels_with_tokens(labels, word_ids, context_len))
    tokenized_inputs["labels"] = new_labels
    
    return tokenized_inputs

def prepare_compute_metrics(id2tags):
    def compute_metrics(eval_preds):
        nonlocal id2tags
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[id2tags[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id2tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        results = {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
        for label, v in all_metrics.items():
            if isinstance(v, dict):
                for m, score in v.items():
                    results[f"{label}_{m}"] = score
        return results
    
    return compute_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="bert-large-cased",
        type=str,
        help="Model for training",
    )
    parser.add_argument(
        "--data_dir", default="./data/samsum/", type=str, help="Input data dir"
    )
    parser.add_argument(
        "--label_type", default="labels_sep_bio", type=str, help="Label type"
    )
    parser.add_argument(
        "--model_dir", default="./model/span-predictor/", type=str, help="Model output dir"
    )

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    new_dataset = load_from_disk(args.data_dir)
    model_id = "-".join(args.data_dir.split("/")[-2].split("-")[2:])
    label_key = args.label_type
    model_type = args.model_name_or_path

    label2id = label_map[label_key]["label2id"]
    id2label = label_map[label_key]["id2label"]
    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_type, id2label=id2label, label2id=label2id)

    tokenized_datasets = new_dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=new_dataset["train"].column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_key": label_key,
        },
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = prepare_compute_metrics(id2label)

    args = TrainingArguments(
        output_dir=args.model_dir,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"./logs/span-predictor/{model_type}-{model_id}-{label_key}",
        logging_strategy="steps",
        logging_steps=500,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    logger.info("*** Training ***")
    logger.info(f"Current model_id: {model_id}")
    logger.info(f"Current label_key: {label_key}")
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_datasets["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(tokenized_datasets["validation"])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    logger.info("*** Predict ***")
    predict_results = trainer.predict(
        tokenized_datasets["test"],
        metric_key_prefix="predict")

    metrics = predict_results.metrics
    metrics["predict_samples"] = len(tokenized_datasets["test"])

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    torch.cuda.empty_cache()
