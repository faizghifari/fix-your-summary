import random

from utils.helper import nlp, update_labels

def replace_noun(summary, labels, distortion_map, alpha=0.5, **kwargs):
    # Extract nouns from the dialog
    dialog = kwargs["dialog"]
    dialog_doc = nlp(dialog)
    dialog_nouns = [token.text for token in dialog_doc if token.pos_ == "NOUN" and not token.ent_type_ and "#" not in token.text]
    dialog_nouns = list(set(dialog_nouns))
    
    summary_doc = nlp(summary)
    summary_nouns = [token for token in summary_doc if token.pos_ == "NOUN" and not token.ent_type_ and "#" not in token.text]

    if len(summary_nouns) == 0:
        return summary, labels, distortion_map
    
    candidates = dialog_nouns + [token.text for token in summary_nouns]

    num_noun_to_replace = max(round(alpha * len(summary_nouns)), 1)
    summary_nouns = random.sample(summary_nouns, num_noun_to_replace)

    # Replace nouns in the summary if only there are more than 1 noun in the dialogue
    if len(candidates) > 1:
        for token in reversed(summary_doc):
            if token.pos_ == "NOUN" and not token.ent_type_ and token in summary_nouns:
                noun = token
                new_candidates = [n for n in candidates if n.lower() != noun.text.lower()]
                if len(new_candidates) > 1:
                    new_noun = random.choice(new_candidates)
                    while new_noun == noun.text.lower():
                        new_noun = random.choice(new_candidates)
                    
                    distortion_map.append({
                        "start_idx": noun.i,
                        "end_idx": noun.i + 1,
                        "correct_tokens": noun.text,
                        "distorted_tokens": new_noun,
                        "type": "N"
                    })
                    labels = update_labels(labels, "N", noun.i, noun.i + 1, new_noun)
                    summary = summary[:noun.idx] + new_noun + summary[noun.idx + len(noun):]

    return summary, labels, distortion_map