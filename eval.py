import evaluate
import json
import pandas as pd
import nltk


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
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    sacrebleu = evaluate.load("sacrebleu")
    origins = json.load(open('/home/palm/data/coco/annotations/annotations/caption_human_thai_val2017.json'))
    df = pd.read_csv('outputs.csv')
    gt = {}
    for ann in origins['annotations']:
        if int(ann['image_id']) not in gt:
            gt[int(ann['image_id'])] = []
        gt[int(ann['image_id'])].append(ann['caption_thai'])
    pd = {}
    for idx, row in df.iterrows():
        key = int(row[0][:-4])
        if key in gt:
            pd[key] = row[1]
    labels = []
    preds = []
    for key in gt:
        if len(gt[key]) < 4:
            continue
        elif len(gt[key]) > 4:
            gt[key] = gt[key][:4]
        labels.append(gt[key])
        preds.append(pd[key])
    compute_metrics((preds, labels))
    results = sacrebleu.compute(predictions=preds,
                                references=labels)