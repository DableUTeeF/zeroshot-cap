import pandas as pd
import json
import nltk
from transformers import AutoTokenizer
import evaluate


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


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

    llama0 = pd.read_csv('llama_output_0.csv')
    llama1 = pd.read_csv('llama_output_1.csv')
    grammared = {}
    tled = {}
    for i in range(len(llama0)):
        for row in (llama0.iloc[i], llama1.iloc[i]):
            if not isinstance(row['grammared'], float):
                grammared[int(row[0][:-4])] = ' '.join([tokenizer.decode(x) for x in tokenizer(row['grammared'])['input_ids'][1:-1]])
                tled[int(row[0][:-4])] = ' '.join([tokenizer.decode(x) for x in tokenizer(row['tled'])['input_ids'][1:-1]])
                break

    for dic in (grammared, tled):
        labels = []
        preds = []
        for key in gt:
            if len(gt[key]) < 2:
                continue
            elif len(gt[key]) > 2:
                gt[key] = gt[key][:2]
            labels.append(gt[key])
            preds.append(dic[key])
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

