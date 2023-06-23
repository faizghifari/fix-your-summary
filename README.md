# Identify-then-Correct Framework for Factual Error Correction
This is the code for our paper titled "Addressing Factual Error in Abstractive Dialogue Summarization via Span Identification and Correction"

## Installation

Create a new python environment and install the requirements

```sh
pip install -r requirements.txt
```

## Corrupted Dataset Generation

To build the corrupted dataset, you can run this command:

```sh
python dataset_gen.py \
    --corrupted_data_ratio 0.5 \
    --max_len 512 \
    --data_dir /path/to/data_dir \
    --output_dir /path/to/output_dir
```

This code will produce a corrupted version of the dataset in HuggingFace dataset format, also with some json data to train identifier, corrector, and baseline FEC model.

Currently this code will only work for dataset with HuggingFace dataset format, specifically with SAMSum and DialogSum datasets. If you want to use other dataset then you need to modify some keys in the code.

The parameter regarding the corruption type is still hard-coded, so if you want to change it please modify it on this line.

```sh
distort_funcs = [
    {"func": replace_entities, "Y": 1, "alpha": 0.5},
    {"func": replace_pronoun, "Y": 1, "alpha": 0.5},
    {"func": replace_quantity, "Y": 1, "alpha": 0.5},
    {"func": replace_noun, "Y": 0.3, "alpha": 0.3},
    {"func": replace_verb, "Y": 0.3, "alpha": 0.3},
]
```

You can also add/remove corruption type by looking at the available corruption type in the `data_gen/` directory.

## Train the Token Identifier Model

To train the token identifier model, you can run this command:

```sh
python train_span_predictor.py \
    --model_name_or_path bert-large-uncased \
    --data_dir /path/to/corrupted_data_dir \
    --model_dir /path/to/model_output_dir \
    --label_type labels_sep_bio
```

The `label_type` arg is for different corruption label type and tagging format. The default value is `labels_sep_bio` which separate the corruption label and use BIO tagging format. Other available value is `labels_sep, labels_bio, labels`. No `sep` in the value means the corruption label is combined, and no `bio` in the value means did not use the BIO tagging format.

## Train the Joint Identifier Model

To train the joint identifier model, you can run this command:

```sh
python train_jointbert.py \
    --do_train \
    --do_eval \
    --do_predict \
    --model_name_or_path bert-large-uncased \
    --data_dir /path/to/corrupted_data_dir \
    --model_dir /path/to/model_output_dir \
    --label_type labels_sep_bio
```

You can see other available arguments in the code.

The code `train_jointbert.py` is a re-implementation from [JointBERT](https://github.com/monologg/JointBERT). Please visit the repository for more information about the actual implementation.

## Train the Corrector, Baseline FEC, and vanilla BART

To tran all of this Seq2Seq model, we utilize the code provided by HuggingFace transformers. You can run the command bellow to train all of the models by utilizing different data, which should be provided if you generate the corrupted dataset using the given command above.

```sh
python run_summarization.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file /path/to/train_json_file \
    --validation_file /path/to/validation_json_file \
    --test_file /path/to/test_json_file \
    --text_column src \
    --summary_column tgt \
    --output_dir /path/to/model_output_dir \
    --label_smoothing_factor 0.1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --predict_with_generate \
    --save_total_limit 2 \
    --load_best_model_at_end
```

You can adjust the data and arguments as you want.