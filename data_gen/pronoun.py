import random

from utils.helper import nlp, update_labels

pronoun_groups = {
    "subject_pronoun": ["he", "she", "it", "they"],
    "possessive_pronoun": ["his", "hers", "theirs"],
    "possessive_adj": ["his", "her", "its", "their"],
    "objective_pron": ["him", "her", "it", "them"],
}
all_pronouns = list(set().union(*pronoun_groups.values()))

def replace_pronoun(summary, labels, distortion_map, alpha=0.5, **kwargs):
    summary_doc = nlp(summary)
    summary_pronouns = [token for token in summary_doc if token.pos_ == "PRON" and token.text.lower() in all_pronouns]
    
    if len(summary_pronouns) == 0:
        return summary, labels, distortion_map
    
    num_pronoun_to_replace = max(round(alpha * len(summary_pronouns)), 1)
    summary_pronouns = random.sample(summary_pronouns, num_pronoun_to_replace)
    
    rev_summary_doc = [token for token in summary_doc]
    rev_summary_doc.reverse()
    for idx, token in enumerate(reversed(summary_doc)):
        if token.pos_ == "PRON" and token in summary_pronouns:
            pronoun = token
            pronoun_group = []
            for group, pronouns in pronoun_groups.items():
                if pronoun.text.lower() in pronouns:
                    pronoun_group.append(group)
            
            if len(pronoun_group) == 1:
                candidates = pronoun_groups[pronoun_group[0]]
            elif len(pronoun_group) > 1:
                if pronoun.text.lower() == "his":
                    if any(not c.isalnum() for c in rev_summary_doc[idx-1].text.lower()):
                        pronoun_group = "possessive_pronoun"
                    else:
                        pronoun_group = "possessive_adj"
                elif pronoun.text.lower() == "her":
                    if any(not c.isalnum() for c in rev_summary_doc[idx-1].text.lower()):
                        pronoun_group = "objective_pron"
                    else:
                        pronoun_group = "possessive_adj"
                elif pronoun.text.lower() == "it":
                    if any(not c.isalnum() for c in rev_summary_doc[idx-1].text.lower()):
                        pronoun_group = "objective_pron"
                    else:
                        pronoun_group = "subject_pronoun"
                else:
                    raise Exception()
                candidates = pronoun_groups[pronoun_group]
            else:
                continue

            new_candidates = [p for p in candidates if p.lower() != pronoun.text.lower()]
            if len(new_candidates) > 1:
                new_pronoun = random.choice(new_candidates)
                while new_pronoun.lower() == pronoun.text.lower():
                    new_pronoun = random.choice(new_candidates)
            else:
                raise Exception()
            
            if pronoun.text[0].isupper():
                new_pronoun = new_pronoun.capitalize()
                
            distortion_map.append({
                "start_idx": pronoun.i,
                "end_idx": pronoun.i + 1,
                "correct_tokens": pronoun.text,
                "distorted_tokens": new_pronoun,
                "type": "P"
            })
            labels = update_labels(labels, "P", pronoun.i, pronoun.i + 1, new_pronoun)
            summary = summary[:pronoun.idx] + new_pronoun + summary[pronoun.idx + len(pronoun):]
    
    return summary, labels, distortion_map
