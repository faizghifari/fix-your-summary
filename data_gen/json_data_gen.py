import os
import json

from utils.labels_tags import add_tags_to_summary_bio

weak_prompt = "Given a dialogue context, a draft summary, a list of potential factually incorrect words/spans, produce a faithful final summary based on the dialogue context."

def generate_json_data(output_dir, dataset):
    json_dir = f"{output_dir}/json/"
    os.makedirs(json_dir, exist_ok=True)

    # create json data to train generative identifier
    identifier_dir = f"{json_dir}/identifier/"
    os.makedirs(identifier_dir, exist_ok=True)

    # create json data to train baseline fec corrector
    fec_dir = f"{json_dir}/baseline_fec/"
    os.makedirs(fec_dir, exist_ok=True)

    # create json data to train tag_sep corrector
    tag_sep_dir = f"{json_dir}/tag_sep/"
    os.makedirs(tag_sep_dir, exist_ok=True)

    # create json data to train tag_comb corrector
    tag_comb_dir = f"{json_dir}/tag_comb/"
    os.makedirs(tag_comb_dir, exist_ok=True)

    # create json data to train list corrector
    list_dir = f"{json_dir}/list/"
    os.makedirs(list_dir, exist_ok=True)

    for split in dataset.keys():
        with open(f"{identifier_dir}/{split}_list.json", "w") as idf:
            with open(f"{identifier_dir}/{split}_tag_sep.json", "w") as idg:
                with open(f"{identifier_dir}/{split}_tag_comb.json", "w") as idh:
                    with open(f"{fec_dir}/{split}.json", "w") as bf:
                        with open(f"{tag_sep_dir}/{split}.json", "w") as tsf:
                            with open(f"{tag_comb_dir}/{split}.json", "w") as tcf:
                                with open(f"{list_dir}/{split}.json", "w") as lf:
                                    for data in dataset[split]:
                                        dialogue = " ".join(data["dialogues"])
                                        distorted_summaries = " ".join(data["distorted_summaries"])
                                        
                                        summary_with_tags_sep = add_tags_to_summary_bio(data["distorted_summaries"], data["labels_sep_bio"], "labels_sep_bio")
                                        summary_with_tags_comb = add_tags_to_summary_bio(data["distorted_summaries"], data["labels_bio"], "labels_bio")

                                        distortion_map = data["distortion_maps"]
                                        word_list = [dist_map["distorted_tokens"] for dist_map in distortion_map]
                                        word_list = "; ".join(word_list) + ";"
                                        input_str = f"{dialogue} </s></s> {distorted_summaries}"

                                        list_json_data = {
                                            "src": input_str,
                                            "tgt": word_list,
                                        }
                                        print(json.dumps(list_json_data), file=idf)

                                        tag_sep_json_data = {
                                            "src": input_str,
                                            "tgt": summary_with_tags_sep,
                                        }
                                        print(json.dumps(tag_sep_json_data), file=idg)

                                        tag_comb_json_data = {
                                            "src": input_str,
                                            "tgt": summary_with_tags_comb,
                                        }
                                        print(json.dumps(tag_comb_json_data), file=idh)

                                        fec_input_str = f"{dialogue} </s></s> {distorted_summaries}"
                                        fec_json_data = {
                                            "src": fec_input_str,
                                            "tgt": data["ref_summaries"],
                                        }
                                        print(json.dumps(fec_json_data), file=bf)

                                        if not all(p == 0 for p in data["labels_sep_bio"]):
                                            tag_sep_input_str = f"{dialogue} </s></s> {summary_with_tags_sep}"
                                            tag_sep_json_data = {
                                                "src": tag_sep_input_str,
                                                "tgt": data["ref_summaries"],
                                            }
                                            print(json.dumps(tag_sep_json_data), file=tsf)

                                            tag_comb_input_str = f"{dialogue} </s></s> {summary_with_tags_comb}"
                                            tag_comb_json_data = {
                                                "src": tag_comb_input_str,
                                                "tgt": data["ref_summaries"],
                                            }
                                            print(json.dumps(tag_comb_json_data), file=tcf)

                                            list_input_str = f"{weak_prompt} </s></s> Word List: {word_list} </s></s> Draft Summary: {distorted_summaries} </s></s> Dialogue Context: {dialogue}"
                                            list_json_data = {
                                                "src": list_input_str,
                                                "tgt": data["ref_summaries"],
                                            }
                                            print(json.dumps(list_json_data), file=lf)
