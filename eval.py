import evaluate
import json
import pandas as pd
import nltk
from transformers import AutoTokenizer


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    decoded_preds, decoded_labels = eval_preds
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels,
                                 use_stemmer=True)
    bleu_result = bleu.compute(predictions=decoded_preds,
                               references=decoded_labels)
    return rouge_result, bleu_result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tha_Thai")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    sacrebleu = evaluate.load("sacrebleu")
    origins = json.load(open('/home/palm/data/coco/annotations/annotations/caption_human_thai_val2017.json'))
    gt = {}
    for ann in origins['annotations']:
        if int(ann['image_id']) not in gt:
            gt[int(ann['image_id'])] = []
        text = ' '.join([tokenizer.decode(x) for x in tokenizer(ann['caption_thai'])['input_ids'][1:-1]])
        gt[int(ann['image_id'])].append(text)
    pds = json.load(open('outputs3.json'))
    pd = {}
    for k in pds:
        key = int(k[:-4])
        if key in gt:
            cap = pds[k]
            text = ' '.join([tokenizer.decode(x) for x in tokenizer(cap[0])['input_ids'][1:-1]])
            pd[key] = text
    labels = []
    preds = []
    for key in gt:
        if len(gt[key]) < 2:
            continue
        elif len(gt[key]) > 2:
            gt[key] = gt[key][:2]
        labels.append(gt[key])
        preds.append(pd[key])
    results = sacrebleu.compute(predictions=preds,
                                references=labels)
    print('sacrebleu')
    print(results)
    rouge_result = rouge.compute(predictions=preds,
                                 references=labels,
                                 use_stemmer=True)
    print('rouge_result')
    print(rouge_result)
    meteor_result = meteor.compute(predictions=preds,
                                   references=labels)
    print('meteor')
    print(meteor_result)
