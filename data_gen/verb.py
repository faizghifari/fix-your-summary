import random

from utils.helper import nlp, update_labels

verb_form = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

modal_groups = {
        "shall": ["shall", "should"],
        "will": ["will", "'ll", "would"],
        "can": ["can", "could"],
        "may": ["may", "might"],
        "must": ["must"]
    }
modals = ["shall", "should", "can", "could", "will", "would", "may", "must", "might"]
all_modals = list(set(modals + [modal for group in modal_groups.values() for modal in group]))

def replace_verb(summary, labels, distortion_map, alpha=0.5, **kwargs):
    # Extract verbs from the dialog
    dialog = kwargs["dialog"]
    dialog_doc = nlp(dialog)
    dialog_verbs = [token.text for token in dialog_doc if token.pos_ == "VERB"]
    dialog_verbs = list(set(dialog_verbs))

    summary_doc = nlp(summary)
    summary_verbs = [token for idx, token in enumerate(summary_doc) if token.pos_ == "VERB"]

    if len(summary_verbs) == 0:
        return summary, labels, distortion_map
    
    candidates = dialog_verbs + [token.text for token in summary_verbs]

    num_verb_to_replace = max(round(alpha * len(summary_verbs)), 1)
    summary_verbs = random.sample(summary_verbs, num_verb_to_replace)

    # Replace verbs in the summary if only there are more than 1 verb in the dialogue
    if len(candidates) > 1:
        for token in reversed(summary_doc):
            if token.pos_ == "VERB" and token in summary_verbs:
                verb = token
                new_candidates = [v for v in candidates if v.lower() != verb.text.lower()]
                if len(new_candidates) > 1:
                    new_verb = random.choice(new_candidates)
                    while new_verb == verb.text.lower():
                        new_verb = random.choice(new_candidates)
                    
                    distortion_map.append({
                        "start_idx": verb.i,
                        "end_idx": verb.i + 1,
                        "correct_tokens": verb.text,
                        "distorted_tokens": new_verb,
                        "type": "V"
                    })
                    labels = update_labels(labels, "V", verb.i, verb.i + 1, new_verb)
                    summary = summary[:verb.idx] + new_verb + summary[verb.idx + len(verb):]

    return summary, labels, distortion_map

def replace_verb_form(summary, labels, distortion_map, alpha=0.5, **kwargs):
    summary_doc = nlp(summary)
    summary_verbs = [token for idx, token in enumerate(summary_doc) if token.pos_ == "VERB"]

    if len(summary_verbs) == 0:
        return summary, labels, distortion_map
    
    num_verb_to_replace = max(round(alpha * len(summary_verbs)), 1)
    summary_verbs = random.sample(summary_verbs, num_verb_to_replace)

    for token in reversed(summary_doc):
        if token.pos_ == "VERB" and token in summary_verbs:
            verb = token
            
            new_form = random.choice(verb_form)
            new_verb = token._.inflect(new_form)
            if new_verb is None:
                continue
            while new_verb == verb.text.lower():
                new_form = random.choice(verb_form)
                new_verb = token._.inflect(new_form)
            
            distortion_map.append({
                "start_idx": verb.i,
                "end_idx": verb.i + 1,
                "correct_tokens": verb.text,
                "distorted_tokens": new_verb,
                "type": "V"
            })
            labels = update_labels(labels, "V", verb.i, verb.i + 1, new_verb)
            summary = summary[:verb.idx] + new_verb + summary[verb.idx + len(verb):]
    
    return summary, labels, distortion_map

def replace_modal(summary, labels, distortion_map, alpha=0.5, **kwargs):
    summary_doc = nlp(summary)
    summary_modals = [token for token in summary_doc if token.pos_ == "MD" or token.text == "'ll"]

    if len(summary_modals) == 0:
        return summary, labels, distortion_map
    
    num_modal_to_replace = max(round(alpha * len(summary_modals)), 1)
    summary_modals = random.sample(summary_modals, num_modal_to_replace)
    
    for token in reversed(summary_doc):
        if token.tag_ == "MD" and token in summary_modals:
            modal = token
            modal_group = None
            # Find the group that this modal belongs to
            for group, modals in modal_groups.items():
                if modal.text.lower() in modals:
                    modal_group = group
                    break
            
            new_modal = random.choice([m for m in all_modals if m not in modal_groups.get(modal_group, [])])
            while new_modal == modal.text.lower():
                new_modal = random.choice(modals)
            
            distortion_map.append({
                "start_idx": modal.i,
                "end_idx": modal.i + 1,
                "correct_tokens": modal.text,
                "distorted_tokens": new_modal,
                "type": "V"
            })
            labels = update_labels(labels, "V", modal.i, modal.i + 1, new_modal)
            summary = summary[:modal.idx] + new_modal + summary[modal.idx + len(modal):]

    return summary, labels, distortion_map