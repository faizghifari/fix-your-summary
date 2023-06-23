label_tags_sep_bio = ["O", "B-E", "I-E", "B-P", "I-P", "B-V", "I-V", "B-N", "I-N", "B-Q", "I-Q"]
label_tags_sep = ["O", "E", "P", "V", "N", "Q"]
label_tags_bio = ["O", "B-D", "I-D"]
label_tags = ["O", "D"]

label_map = {
    "labels_sep_bio": {
        "tags": label_tags_sep_bio,
        "label2id": {label: i for i, label in enumerate(label_tags_sep_bio)},
        "id2label": {i: label for i, label in enumerate(label_tags_sep_bio)},
    },
    "labels_sep": {
        "tags": label_tags_sep,
        "label2id": {label: i for i, label in enumerate(label_tags_sep)},
        "id2label": {i: label for i, label in enumerate(label_tags_sep)},
    },
    "labels_bio": {
        "tags": label_tags_bio,
        "label2id": {label: i for i, label in enumerate(label_tags_bio)},
        "id2label": {i: label for i, label in enumerate(label_tags_bio)},
    },
    "labels": {
        "tags": label_tags,
        "label2id": {label: i for i, label in enumerate(label_tags)},
        "id2label": {i: label for i, label in enumerate(label_tags)},
    },
}

def convert_labels_remove_bio(labels):
    new_labels = []
    for label in labels:
        if label == "O":
            new_labels.append("O")
        else:
            new_label = label.split("-")[1]
            new_labels.append(new_label)
    return new_labels

def convert_labels_to_one_type(labels):
    new_labels = []
    current_label = None
    for label in labels:
        if label == "O":
            new_labels.append("O")
            current_label = "O"
        else:
            if current_label == "O" or current_label is None:
                new_label = "B-D"
            else:
                new_label = "I-D"
            new_labels.append(new_label)
            current_label = new_label
    return new_labels

def add_tags_to_summary_bio(summary, labels, label_key):
    tags = label_map[label_key]["tags"]
    tags2id = label_map[label_key]["label2id"]
    # Create a mapping from label id to tag
    id2tag = {tags2id[label]: f"<{label[2:].lower()}>" for label in tags if label != "O"}

    # Initialize the current label to 'O'
    current_label = "O"

    # Iterate over the tokens and labels
    tokens_with_tags = []
    for token, label in zip(summary, labels):
        # If the label is 'O', add the token to the output list
        if label == 0:
            if current_label != "O":
                tokens_with_tags[-1] += f"</{current_label[2:].lower()}>"
            tokens_with_tags.append(token)
            current_label = "O"
        else:
            # If the label is not 'O', get the corresponding tag
            tag = id2tag[label]

            # If the label is a beginning label (B-*)
            if tags[label].startswith("B-"):
                # If the current label is not 'O', close the previous tag
                if current_label != "O":
                    tokens_with_tags[-1] += f"</{current_label[2:].lower()}>"

                # Add the opening tag to the token
                tokens_with_tags.append(f"{tag}{token}")
                current_label = tags[label]
            # If the label is an inside label (I-*)
            elif tags[label].startswith("I-"):
                # If the current label is 'O', treat it as a beginning label
                if current_label == "O":
                    tokens_with_tags.append(f"{tag}{token}")
                    current_label = tags[label]
                # If the current label matches the label, add the token to the current tag
                elif current_label[2:] == tags[label][2:]:
                    tokens_with_tags[-1] += f" {token}"
                # If the current label doesn't match the label, close the previous tag and start a new tag
                else:
                    tokens_with_tags[-1] += f"</{current_label[2:].lower()}> {tag}{token}"
                    current_label = tags[label]

    # If the last token was part of a tag, close the tag
    if current_label != "O":
        tokens_with_tags[-1] += f"</{current_label[2:].lower()}>"

    # Join the tokens into a single string
    summary_with_tags = " ".join(tokens_with_tags)

    return summary_with_tags

def add_tags_to_summary(summary, labels, label_key):
    tags = label_map[label_key]["tags"]
    tags2id = label_map[label_key]["label2id"]
    # Create a mapping from label id to tag
    id2tag = {tags2id[label]: f"<{label.lower()}>" for label in tags if label != "O"}

    # Initialize the current label to 'O'
    current_label = "O"

    # Iterate over the tokens and labels
    tokens_with_tags = []
    for token, label in zip(summary, labels):
        # If the label is 'O', add the token to the output list
        if label == 0:
            if current_label != "O":
                tokens_with_tags[-1] += f"</{current_label.lower()}>"
            tokens_with_tags.append(token)
            current_label = "O"
        else:
            # If the label is not 'O', get the corresponding tag
            tag = id2tag[label]
            # If the current label is 'O', treat it as a beginning label
            if current_label == "O":
                tokens_with_tags.append(f"{tag}{token}")
                current_label = tags[label]
            # If the current label matches the label, add the token to the current tag
            elif current_label == tags[label]:
                tokens_with_tags[-1] += f" {token}"
            # If the current label doesn't match the label, close the previous tag and start a new tag
            else:
                tokens_with_tags[-1] += f"</{current_label.lower()}> {tag}{token}"
                current_label = tags[label]

    # If the last token was part of a tag, close the tag
    if current_label != "O":
        tokens_with_tags[-1] += f"</{current_label.lower()}>"

    # Join the tokens into a single string
    summary_with_tags = " ".join(tokens_with_tags)

    return summary_with_tags
