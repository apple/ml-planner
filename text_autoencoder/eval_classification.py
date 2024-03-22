#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from interpolation import parse_args, load_model
import numpy as np
import tqdm
import re
from text_autoencoder.baseline.classifier_text import BERTClassifier
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset



def eval_classification(args):
    with open(args.input, "r") as f:
        input_lines =  [line.strip() for line in f.readlines()]
            
    # eval acc via BertClassier
    device = torch.device("cuda")
    classifier = BERTClassifier.from_pretrained("", args.classifier_path)
    classifier = classifier.cuda()
    classifier.eval()
    label = torch.LongTensor([args.cond]*len(input_lines))
    inputs = classifier.tokenizer.batch_encode_plus(input_lines, pad_to_max_length=True)
    input_ids = torch.LongTensor(inputs['input_ids'])
    dataset = TensorDataset(input_ids, label)
    dataloader = DataLoader(dataset, batch_size=64)
    correct = 0
    total_predictions = []
    for batch in dataloader:
    
        batch = tuple(t.to(device) for t in batch)

        _, corr, prediction = classifier(batch[0], batch[1])
        correct += corr
        total_predictions.extend(prediction.tolist())

    with open(args.output, "a") as f:
        f.write(f"Gen ACC: {correct / len(input_lines)}\n")
        f.write(f"Predictions: {total_predictions}\n")


if __name__ == "__main__":
    parser = parse_args()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--classifier_path", type=str, default="models/classifier/epoch50-step664999-acc0.962")
    parser.add_argument('--cond', type=int, default=None, help='conditional value')
    args = parser.parse_args()
    eval_classification(args)