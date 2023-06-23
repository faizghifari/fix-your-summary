from utils.helper import nlp

def find_matching_token(list_a, list_b, max_range=5):
    switch = False
    if len(list_a) > len(list_b):
        list_a, list_b = list_b, list_a
        switch = True
    
    for idx_a in range(min(len(list_a), max_range)):
        token_a = list_a[idx_a]
        if token_a in list_b:
            idx_b = list_b.index(token_a)
            if abs(idx_b - idx_a) <= max_range:
                if switch:
                    return idx_b, idx_a
                else:
                    return idx_a, idx_b
                
    return None, None

def find_different_tokens_with_indexes(pred, distorted):
    switch = False
    if len(pred) > len(distorted):
        pred, distorted = distorted, pred
        switch = True

    results = []
    dist_idx = 0
    pred_idx = 0
    while pred_idx < len(pred) and dist_idx < len(distorted):
        if pred[pred_idx] == distorted[dist_idx]:
            dist_idx += 1
            pred_idx += 1
            continue

        start_idx_pred = pred_idx
        start_idx_distorted = dist_idx

        end_idx_pred = None
        end_idx_distorted = None

        pred_list = pred[pred_idx+1:]
        distorted_list = distorted[dist_idx+1:]
        if pred_list and distorted_list:
            end_pred, end_dist = find_matching_token(pred_list, distorted_list)
            if end_pred is not None:
                end_idx_pred = pred_idx + end_pred + 1
                end_idx_distorted = dist_idx + end_dist + 1

        if end_idx_pred is None or end_idx_distorted is None:
            end_idx_pred = pred_idx + 1
            end_idx_distorted = dist_idx + 1

        token_pred = " ".join(pred[start_idx_pred:end_idx_pred])
        token_distorted = " ".join(distorted[start_idx_distorted:end_idx_distorted])

        if switch:
            results.append({
                "token_pred": token_distorted,
                "start_idx_pred": start_idx_distorted,
                "end_idx_pred": end_idx_distorted,
                "token_distorted": token_pred,
                "start_idx_distorted": start_idx_pred,
                "end_idx_distorted": end_idx_pred,
            })
        else:
            results.append({
                "token_pred": token_pred,
                "start_idx_pred": start_idx_pred,
                "end_idx_pred": end_idx_pred,
                "token_distorted": token_distorted,
                "start_idx_distorted": start_idx_distorted,
                "end_idx_distorted": end_idx_distorted,
            })
        
        pred_idx = end_idx_pred
        dist_idx = end_idx_distorted
    
    results = [dict(t) for t in {tuple(res.items()) for res in results}]
    results = sorted(results, key=lambda d: d["start_idx_distorted"])

    return results

def count_correct_pred_tokens(diff, distort_map, soft=False):
    num_corrected_tokens = 0
    if len(distort_map["correct_tokens"].split()) == 1 and len(diff["token_pred"].split()) == 1:
        if soft:
            if distort_map["correct_tokens"].lower() == diff["token_pred"].lower():
                num_corrected_tokens += distort_map["end_idx"] - distort_map["start_idx"]
        else:
            if distort_map["correct_tokens"] == diff["token_pred"]:
                num_corrected_tokens += distort_map["end_idx"] - distort_map["start_idx"]
    else:
        if soft:
            if distort_map["correct_tokens"].lower() in diff["token_pred"].lower():
                num_corrected_tokens += distort_map["end_idx"] - distort_map["start_idx"]
        else:
            if distort_map["correct_tokens"] in diff["token_pred"]:
                num_corrected_tokens += distort_map["end_idx"] - distort_map["start_idx"]
    return num_corrected_tokens

def filter_distortion_maps(data):
    ref = " ".join([token.text for token in nlp(data["ref_summaries"])])
    ref = " ".join([token.text for token in nlp(ref)])
    ref_tokens = [token.text for token in nlp(ref)]
    distorted_summaries = data["distorted_summaries"]
    distortion_maps = data["distortion_maps"]

    # Step 1: Check if distorted_tokens exist in distorted_summaries
    distortion_maps = [d for d in distortion_maps if d["distorted_tokens"] in distorted_summaries[d["start_idx"]:d["end_idx"]] and d["distorted_tokens"] not in ref_tokens[d["start_idx"]:d["end_idx"]]]

    seen_distortions = {}
    filtered_maps = []
    for d in distortion_maps:
        key = (d["start_idx"], d["end_idx"], d["distorted_tokens"])
        if key in seen_distortions:
            filtered_maps.remove(seen_distortions[key])  # Remove previous map
        seen_distortions[key] = d  # Update seen_distortions with the new map
        filtered_maps.append(d)

    return filtered_maps

def compute_correction_ratio(data, pred_summary, soft=False, max_range=3):
    pred = " ".join([token.text for token in nlp(pred_summary)])
    pred = " ".join([token.text for token in nlp(pred)])
    pred = " ".join([token.text for token in nlp(pred)])

    distortion_map = filter_distortion_maps(data)
    distortion_map = sorted(distortion_map, key=lambda d: d['start_idx']) 
    num_distorted_tokens = sum([d["end_idx"] - d["start_idx"] for d in distortion_map])
    
    if num_distorted_tokens == 0:
        return 0, 0

    num_corrected_tokens = 0
    diff_tokens = find_different_tokens_with_indexes(pred.split(), data["distorted_summaries"])
    for d in distortion_map:
        diff_candidates = [diff for diff in diff_tokens if diff["start_idx_distorted"] == d["start_idx"] and d["distorted_tokens"] in diff["token_distorted"]]
        if not diff_candidates:
            diff_candidates = [diff for diff in diff_tokens if d["distorted_tokens"] in diff["token_distorted"]]
            if not diff_candidates:
                continue
            else:
                if len(diff_candidates) == 1:
                    num_corrected_tokens += count_correct_pred_tokens(diff_candidates[0], d, soft=soft)
                else:
                    diff_candidates = [diff for diff in diff_candidates if abs(diff["start_idx_distorted"] - d["start_idx"]) <= max_range]
                    if not diff_candidates:
                        continue
                    else:
                        if len(diff_candidates) == 1:
                            num_corrected_tokens += count_correct_pred_tokens(diff_candidates[0], d, soft=soft)
                        else:
                            diff_candidates = [diff for diff in diff_candidates if all(tok in diff["token_distorted"].split() for tok in d["distorted_tokens"].split())]
                            if not diff_candidates:
                                continue
                            else:
                                if len(diff_candidates) == 1:
                                    num_corrected_tokens += count_correct_pred_tokens(diff_candidates[0], d, soft=soft)
                                else:
                                    diff_candidates = [diff for diff in diff_candidates if d["start_idx"] in range(diff["start_idx_distorted"], diff["end_idx_distorted"] + 1)]
                                    if not diff_candidates:
                                        continue
                                    else:
                                        if len(diff_candidates) == 1:
                                            num_corrected_tokens += count_correct_pred_tokens(diff_candidates[0], d, soft=soft)
                                        else:
                                            raise Exception("Error because of multiple candidates in the similar start_idx")
        else:
            if len(diff_candidates) == 1:
                num_corrected_tokens += count_correct_pred_tokens(diff_candidates[0], d, soft=soft)
            else:
                raise Exception("Error because of multiple candidates in the same start_idx")
    
    return num_corrected_tokens, num_distorted_tokens
