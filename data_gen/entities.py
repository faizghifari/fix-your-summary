import random

from utils.helper import nlp, align_distortion_map, update_labels, add_labels

def replace_entities(summary, labels, distortion_map, alpha=0.5, **kwargs):
    dialog = kwargs["dialog"]
    dialog_entities = []
    turns = dialog.split("\n")
    for turn in turns:
        turn.replace("\r", "")
        turn_doc = nlp(turn)
        raw_turn = " ".join([token.text for token in turn_doc])
        turn_doc = nlp(raw_turn)
        dialog_entities.extend([ent for ent in turn_doc.ents if "\r" not in ent.text and not any(char for char in ent.text if not char.isspace() and not char.isalpha())])

    summary_doc = nlp(summary)
    summary_entities = [ent for ent in summary_doc.ents if not any(char for char in ent.text if not char.isspace() and not char.isalpha())]

    if len(summary_entities) == 0:
        return summary, labels, distortion_map

    entity_groups = {}
    for ent in dialog_entities + summary_entities:
        if ent.label_ not in entity_groups.keys():
            entity_groups[ent.label_] = [ent.text]
        else:
            entity_groups[ent.label_].append(ent.text)
    
    for group in entity_groups.keys():
        entity_groups[group] = list(set(entity_groups[group]))

    summary_entities = [ent for ent in summary_entities if len(entity_groups[ent.label_]) > 1]

    if len(summary_entities) == 0:
        return summary, labels, distortion_map
    
    num_entities_to_replace = max(round(alpha * len(summary_entities)), 1)
    summary_entities = random.sample(summary_entities, num_entities_to_replace)
    summary_entities.sort(key=lambda ent: ent.start)
    
    options = ["replace", "replace", "insert"]
    for entity in reversed(summary_entities):
        lower_idx = max(entity.start - 3, 0)
        upper_idx = min(entity.end + 3, len(summary_doc) - 1)
        prohibited_candidates = summary_doc[lower_idx:upper_idx].text
        new_candidates = entity_groups[entity.label_]
        new_candidates = list(set([ent for ent in new_candidates if ent.lower() != entity.text.lower() and ent.lower() not in entity.text.lower() and entity.text.lower() not in ent.lower() and ent not in prohibited_candidates]))
        if len(new_candidates) > 0:
            new_entity = random.choice(new_candidates)
            while new_entity.lower() == entity.text.lower():
                new_entity = random.choice(new_candidates)
        else:
            continue

        option = random.choice(options)
        if entity.label_ == "PERSON" and option == "insert":
            lower_idx = max(entity.start - 1, 0)
            upper_idx = entity.end
            if summary_doc[lower_idx].text == "," and summary_doc[upper_idx].text == ",":
                input_str = f" , {new_entity}"
            else:
                input_str = f" and {new_entity}"
            
            distortion_map = align_distortion_map(distortion_map, entity.end, 1 + len(new_entity.split()))
            distortion_map.append({
                "start_idx": entity.start + 2,
                "end_idx": entity.start + 2 + len(new_entity.split()),
                "correct_tokens": "",
                "distorted_tokens": new_entity,
                "type": "E"
            })
            labels = add_labels(labels, "E", entity.end, new_entity)
            summary = summary[:entity.end_char] + input_str + summary[entity.end_char:]
        else:
            if len(new_entity.split()) > 1:
                distortion_map = align_distortion_map(distortion_map, entity.end, len(new_entity.split()) - 1)
            distortion_map.append({
                "start_idx": entity.start,
                "end_idx": entity.start + len(new_entity.split()),
                "correct_tokens": entity.text,
                "distorted_tokens": new_entity,
                "type": "E"
            })
            labels = update_labels(labels, "E", entity.start, entity.end, new_entity)
            summary = summary[:entity.start_char] + new_entity + summary[entity.end_char:]

    return summary, labels, distortion_map
