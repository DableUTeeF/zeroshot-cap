from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
from datagen import MTLDataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import os
import nltk
import evaluate


def tokenization_fn(captions, max_target_length=120):
    """Run tokenization on captions."""
    labels = tokenizer(captions,
                       padding="max_length",
                       max_length=max_target_length,
                       return_tensors="pt",
                       truncation=True).input_ids

    return labels


def collate_fn(batch):
    model_inputs = {'labels': []}
    inputs = []
    for obj in batch:
        model_inputs['labels'].append(obj[1])
        inputs.append(obj[0])
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])
    outputs = tokenizer(inputs,
                        padding="max_length",
                        max_length=120,
                        return_tensors="pt",
                        truncation=True)
    model_inputs.update(outputs)
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels,
                                 use_stemmer=True)
    return rouge_result


if __name__ == '__main__':
    bs = 2
    output_dir = 'checkpoints/'
    rouge = evaluate.load('rouge')

    model = AutoModelForCausalLM.from_pretrained('clicknext/phayathaibert', is_decoder=True)
    tokenizer = AutoTokenizer.from_pretrained('clicknext/phayathaibert', padding_side='left')

    train_set = MTLDataset('data/train2.csv')
    valid_set = MTLDataset('data/val2.csv')

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=12,
        output_dir=os.path.join(output_dir),
        logging_dir='logs',
        dataloader_num_workers=0,
        logging_strategy='steps',
        logging_steps=10,
        # disable_tqdm=True,
        generation_max_length=124,
        report_to=['tensorboard']
    )
    trainer = Seq2SeqTrainer(
        model=model,
        # tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate_fn,
    )
    trainer.train()
