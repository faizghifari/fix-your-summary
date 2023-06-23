# this is a re-implementation of JointBERT from https://github.com/monologg/JointBERT, please visit the link for more detailed implementation

import os
import sys
import json
import logging
import argparse

import numpy as np
import torch.nn as nn
import evaluate

import torch

from torchcrf import CRF
from tqdm import tqdm, trange

from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup, DataCollatorForTokenClassification, AutoTokenizer

from utils.labels_tags import label_map

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

seqeval = evaluate.load("seqeval")


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.0):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.0):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


def get_token_cls_metric(preds, labels):
    assert len(preds) == len(labels)

    all_metrics = seqeval.compute(predictions=preds, references=labels)
    results = {
        "token_cls_precision": all_metrics["overall_precision"],
        "token_cls_recall": all_metrics["overall_recall"],
        "token_cls_f1": all_metrics["overall_f1"],
        "token_cls_accuracy": all_metrics["overall_accuracy"],
    }
    for label, v in all_metrics.items():
        if isinstance(v, dict):
            for m, score in v.items():
                if isinstance(score, np.integer):
                    score = int(score)
                results[f"{label}_{m}"] = score
    return results


def get_text_cls_metric(preds, labels):
    acc = (preds == labels).mean()
    return {"intent_acc": acc}


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert (
        len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    )
    results = {}

    text_cls_result = get_text_cls_metric(intent_preds, intent_labels)
    token_cls_result = get_token_cls_metric(slot_preds, slot_labels)

    results.update(text_cls_result)
    results.update(token_cls_result)

    return results


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(
            config.hidden_size, self.num_intent_labels, args.dropout_rate
        )
        self.slot_classifier = SlotClassifier(
            config.hidden_size, self.num_slot_labels, args.dropout_rate
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        intent_label_ids,
        slot_labels_ids,
    ):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1), intent_label_ids.view(-1)
                )
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels),
                    intent_label_ids.view(-1),
                )
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                inactive_idx = slot_labels_ids == self.args.ignore_index
                active_labels = slot_labels_ids.masked_fill_(inactive_idx, 0)
                # active_mask = attention_mask.masked_fill_(inactive_idx, 0)
                inactive_idx[:, 0] = False  # prevent error for mask first timestep
                slot_loss = self.crf(
                    slot_logits,
                    active_labels,
                    mask=~inactive_idx,
                    reduction="mean",
                )
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    # active_loss = slot_labels_ids.view(-1) != self.args.ignore_index
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[
                        active_loss
                    ]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(
                        slot_logits.view(-1, self.num_slot_labels),
                        slot_labels_ids.view(-1),
                    )
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = ["clean", "distorted"]
        self.slot_label_lst = label_map[args.label_type]["tags"]
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config = BertConfig.from_pretrained(args.model_name_or_path)
        self.model = JointBERT.from_pretrained(
            args.model_name_or_path,
            config=self.config,
            args=args,
            intent_label_lst=self.intent_label_lst,
            slot_label_lst=self.slot_label_lst,
        )

        # GPU or CPU
        self.device = (
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
        )

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps
                // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "intent_label_ids": batch[3],
                    "slot_labels_ids": batch[4],
                }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if (
                        self.args.logging_steps > 0
                        and global_step % self.args.logging_steps == 0
                    ):
                        self.evaluate("dev")

                    if (
                        self.args.save_steps > 0
                        and global_step % self.args.save_steps == 0
                    ):
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "intent_label_ids": batch[3],
                    "slot_labels_ids": batch[4],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs["intent_label_ids"].detach().cpu().numpy()
            else:
                intent_preds = np.append(
                    intent_preds, intent_logits.detach().cpu().numpy(), axis=0
                )
                out_intent_label_ids = np.append(
                    out_intent_label_ids,
                    inputs["intent_label_ids"].detach().cpu().numpy(),
                    axis=0,
                )

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(
                        slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0
                    )
                else:
                    slot_preds = np.append(
                        slot_preds, slot_logits.detach().cpu().numpy(), axis=0
                    )

                out_slot_labels_ids = np.append(
                    out_slot_labels_ids,
                    inputs["slot_labels_ids"].detach().cpu().numpy(),
                    axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(
                        slot_label_map[out_slot_labels_ids[i][j]]
                    )
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(
            intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list
        )
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        with open(f"{self.args.model_dir}/{mode}_results.json", "w") as f:
            json.dump(results, f)

        return results, intent_preds, slot_preds_list

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = JointBERT.from_pretrained(
                self.args.model_dir,
                args=self.args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")


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
    tokenized_inputs = tokenizer(
        data["dialogues"],
        data["distorted_summaries"],
        is_split_into_words=True,
        truncation=True,
        max_length=512,
    )
    new_labels = []
    hallucination_labels = []
    for i, labels in enumerate(data[label_key]):
        word_ids = tokenized_inputs.word_ids(i)
        dialogue = data["dialogues"][i]
        context_len = len(tokenizer.tokenize(dialogue, is_split_into_words=True))
        new_labels.append(align_labels_with_tokens(labels, word_ids, context_len))

        if all(l == 0 for l in labels):
            hallucination_labels.append(0)
        else:
            hallucination_labels.append(1)
    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["hallucination_label_ids"] = hallucination_labels

    return tokenized_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed for initialization"
    )
    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=16, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout for fully-connected layers",
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predict on the test set.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    parser.add_argument(
        "--ignore_index",
        default=-100,
        type=int,
        help="Specifies a target value that is ignored and does not contribute to the input gradient",
    )

    parser.add_argument(
        "--slot_loss_coef",
        type=float,
        default=1.0,
        help="Coefficient for the slot loss.",
    )

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument(
        "--slot_pad_label",
        default="PAD",
        type=str,
        help="Pad token for slot label pad (to be ignore when calculate loss)",
    )

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
        "--model_dir", default="./model/joint-predictor/", type=str, help="Model output dir"
    )

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    dataset = load_from_disk(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    tokenized_datasets = dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_key": args.label_type,
        },
    )

    tokenized_inputs = {}
    for split in tokenized_datasets.keys():
        tokenized_inputs[split] = data_collator(tokenized_datasets[split])

    train_dataset = TensorDataset(
        tokenized_inputs["train"]["input_ids"],
        tokenized_inputs["train"]["attention_mask"],
        tokenized_inputs["train"]["token_type_ids"],
        tokenized_inputs["train"]["hallucination_label_ids"],
        tokenized_inputs["train"]["labels"],
    )
    validation_dataset = TensorDataset(
        tokenized_inputs["validation"]["input_ids"],
        tokenized_inputs["validation"]["attention_mask"],
        tokenized_inputs["validation"]["token_type_ids"],
        tokenized_inputs["validation"]["hallucination_label_ids"],
        tokenized_inputs["validation"]["labels"],
    )
    test_dataset = TensorDataset(
        tokenized_inputs["test"]["input_ids"],
        tokenized_inputs["test"]["attention_mask"],
        tokenized_inputs["test"]["token_type_ids"],
        tokenized_inputs["test"]["hallucination_label_ids"],
        tokenized_inputs["test"]["labels"],
    )

    trainer = Trainer(args, train_dataset, validation_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("dev")

    if args.do_predict:
        trainer.load_model()
        _, intent_preds, label_preds = trainer.evaluate("test")

        with open(f"{args.model_dir}/test_predict.json", "w") as f:
            for i, intent_pred in enumerate(intent_preds):
                result = {
                    "hallucinated_pred": int(intent_pred),
                    "token_pred": label_preds[i],
                }
                print(json.dumps(result), file=f)
