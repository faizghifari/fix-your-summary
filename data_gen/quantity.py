import random

from utils.helper import nlp, update_labels

def replace_quantity(summary, labels, distortion_map, alpha=0.5, **kwargs):
    dialog = kwargs["dialog"]    
    dialog_doc = nlp(dialog)
    dialog_nums = [token for token in dialog_doc if token.pos_ == "NUM"]
    date_num = list(set([num.text for num in dialog_nums if num.ent_type_ == "DATE"]))
    time_num = list(set([num.text for num in dialog_nums if num.ent_type_ == "TIME"]))
    
    summary_doc = nlp(summary)
    summary_num = [token for token in summary_doc if token.pos_ == "NUM"]

    if len(summary_num) == 0:
        return summary, labels, distortion_map
    
    num_num_to_replace = max(round(alpha * len(summary_num)), 1)
    summary_num = random.sample(summary_num, num_num_to_replace)

    for token in reversed(summary_doc):
        if token.pos_ == "NUM" and token in summary_num and any(char.isdigit() for char in token.text):
            num = token
            if num.ent_type_ == "DATE":
                if len(date_num) > 1:
                    new_num = random.choice(date_num)
                    while new_num == num.text:
                        new_num = random.choice(date_num)
                else:
                    continue
            elif num.ent_type_ == "TIME" or ":" in num.text:
                if len(time_num) > 1:
                    new_num = random.choice(time_num)
                    while new_num == num.text:
                        new_num = random.choice(time_num)
                else:
                    continue
            else:
                continue
            
            distortion_map.append({
                "start_idx": num.i,
                "end_idx": num.i + 1,
                "correct_tokens": num.text,
                "distorted_tokens": new_num,
                "type": "Q"
            })
            labels = update_labels(labels, "Q", num.i, num.i + 1, new_num)
            summary = summary[:num.idx] + new_num + summary[num.idx + len(num):]
    
    return summary, labels, distortion_map