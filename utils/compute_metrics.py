# Copyright 2023-2024 Xiaomi Corporation and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import json
import logging
import os
from re import sub

from utils.eval_captioning import EvalCap


def _text_preprocess(sentence):
    sentence = sentence.lower()
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")
    sentence = sub('[(,.!?;:|*")]', " ", sentence).replace("  ", " ")
    return sentence


def compute_metrics(predict_file, ref_file):
    ref_dict = {}
    with open(ref_file, "r", encoding="utf8") as reader:
        for line in reader:
            obj = json.loads(line)
            name = obj["key"]
            caption = obj["label"].split("\t")
            caption_rex = [_text_preprocess(cap) for cap in caption]
            ref_dict[name] = caption_rex

    predict_dict = {}
    with open(predict_file, "r", encoding="utf8") as reader:
        for line in reader:
            temp = line.strip("\n").split("\t")
            predict_dict[temp[0]] = [temp[1]]

    res_dir = os.path.dirname(predict_file)
    res_prefix = os.path.basename(predict_file).replace(".txt", "")
    eval_scorer = EvalCap(predict_dict, ref_dict)
    metrics = eval_scorer.compute_scores()
    logging.info(
        "meteor {}, cider {}, spice {}, spider {}, spider_fl {}, sentence_bert {}, fense {}".format(
            round(metrics["meteor"], 5),
            round(metrics["cider"], 5),
            round(metrics["spice"], 5),
            round(metrics["spider"], 5),
            round(metrics["spider_fl"], 5),
            round(metrics["sentence_bert"], 5),
            round(metrics["fense"], 5),
        )
    )

    eval_file = os.path.join(
        res_dir,
        "{}_bleu1{}_bleu2{}_bleu3{}_bleu4{}_rougel{}_meteor{}_cider{}_spice{}_spider{}_spiderfl{}_sentence-bert{}_fense{}.csv".format(
            res_prefix,
            round(metrics["bleu_1"], 5),
            round(metrics["bleu_2"], 5),
            round(metrics["bleu_3"], 5),
            round(metrics["bleu_4"], 5),
            round(metrics["rouge_l"], 5),
            round(metrics["meteor"], 5),
            round(metrics["cider"], 5),
            round(metrics["spice"], 5),
            round(metrics["spider"], 5),
            round(metrics["spider_fl"], 5),
            round(metrics["sentence_bert"], 5),
            round(metrics["fense"], 5),
        ),
    )
    with open(eval_file, "w+", encoding="utf8", newline="") as csvfile:
        csv_writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "name",
                "meteor",
                "cider",
                "spice",
                "spider",
                "spider_fl",
                "sentence_bert",
                "fense",
                "error_prob",
                "predicted",
                "Original_1",
                "Original_2",
                "Original_3",
                "Original_4",
                "Original_5",
            ],
        )
        csv_writer.writeheader()
        csv_writer.writerows(metrics["data"])
    logging.info(
        "End eval captioning for {}, {}, ref {}, eval {}".format(
            predict_file, res_prefix, ref_file, eval_file
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test with your special model")
    parser.add_argument("--test_data", required=True, help="test data file")
    parser.add_argument("--predict_dir", required=True, help="predict result file")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(args)

    predict_files = [
        "beam_1",
        "beam_2",
        "beam_3",
        "top-p_0.85",
        "top-p_0.9",
        "top-p_0.95",
    ]
    for pf in predict_files:
        p_file = os.path.join(args.predict_dir, pf + ".txt")
        if os.path.isfile(p_file):
            compute_metrics(p_file, args.test_data)
        p_llm_file = os.path.join(args.predict_dir, pf + "_llm.txt")
        if os.path.isfile(p_llm_file):
            compute_metrics(p_llm_file, args.test_data)
