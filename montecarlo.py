import torch
import os
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
from torch import nn
import numpy as np
import json
import random
from pythainlp import correct


class Model:
    def __init__(self):
        self.transformer = AutoModel.from_pretrained('xlm-roberta-large').eval().to('cuda')
        self.LinearTransformation = torch.nn.Linear(1024, 640).eval().to('cuda')
        self.score_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        d = '/home/palm/.cache/huggingface/hub/models--M-CLIP--XLM-Roberta-Large-Vit-B-16Plus/blobs/a7124266439dd6bce544c23b57249a2ca764dbcd38e8eab16ce272c28c27b049'
        cp = torch.load(d)
        tw = {}
        lw = {}
        for key, value in cp.items():
            if 'transformer' in key:
                key = key.replace('transformer.', '')
                tw[key] = value
            if 'LinearTransformation' in key:
                key = key.replace('LinearTransformation.', '')
                lw[key] = value
        self.transformer.load_state_dict(tw)
        self.LinearTransformation.load_state_dict(lw)
        self.lm_model = AutoModelForMaskedLM.from_pretrained('lst-nectec/HoogBERTa').eval().to('cuda')
        self.lm_tokenizer = AutoTokenizer.from_pretrained('lst-nectec/HoogBERTa')
        self.mask_token_id = self.lm_tokenizer.mask_token_id

    @torch.no_grad()
    def encode(self, text):
        tokens = self.score_tokenizer(text, return_tensors='pt', padding=True, truncation=True).to('cuda')
        embs = self.transformer(**tokens)[0]
        att = tokens['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        embs = self.LinearTransformation(embs)
        embs = embs / embs.norm(dim=-1)
        return embs

    @torch.no_grad()
    def tokenize(self, text):
        return self.lm_tokenizer(text, return_tensors='pt')

    @torch.no_grad()
    def detokenize(self, input_ids):
        return self.lm_tokenizer.decode(input_ids, skip_special_tokens=True).replace(' ', '')

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, token_type_ids):

        return


def get_wl():
    return open('wordlist.txt').read().split('\n')


def search(model,
           prompt='รูปของ<mask><mask><mask><mask><mask><mask><mask><mask>',
           gt='image of a sleeping girl amidst elegant swirling water',
           depth=2,
           width=3,
           samples=100,
           scores=None,
           chosen_index=None,
           index_prop=0.5,
           wordlist=get_wl(),
           ):
    if depth < 1:
        return scores
    encoded_prompt = model.encode(prompt.replace('<mask>', ''))
    encoded_gt = model.encode(gt)
    if scores is None:
        scores = {prompt.replace('<mask>', ''): util.dot_score(encoded_prompt, encoded_gt)[0][0].cpu().numpy().tolist()}
    prompts = model.tokenize(prompt).to('cuda')
    prompt_ids = prompts['input_ids'][0]
    m = model.lm_model(**prompts)
    x = m.logits.argsort(-1, True)
    mask = prompt_ids == model.mask_token_id
    if chosen_index is None:
        if np.random.rand() > index_prop and mask.sum() > 0:
            chosen_idx = torch.where(mask)[0].cpu().numpy()
            idx = np.random.choice(chosen_idx)
        else:
            idx = np.random.randint(x.size(1))
    else:
        idx = chosen_index
    x = x[0, idx, :samples]
    temp_scores = []
    temp_prompts = []
    for s in range(samples):
        temp_token = model.detokenize(x[s])
        if temp_token not in wordlist:
            continue
        temp_input_ids = prompts['input_ids'][0]
        temp_input_ids[idx] = x[s]
        temp_prompt = model.detokenize(temp_input_ids)
        if temp_prompt not in scores:
            feat = model.encode(temp_prompt)
            temp_scores.append(util.dot_score(feat, encoded_gt)[0][0].cpu().numpy().tolist())
            temp_prompts.append(temp_prompt)
    temp_scores = torch.tensor(temp_scores)
    indice = temp_scores.argsort(-1, True)  # change randomizing to here instead
    if np.random.rand() > 0.5:
        if np.random.rand() > 0.5:
            indice = indice.cpu().numpy()
            np.random.shuffle(indice)
        else:
            indice = indice.cpu().numpy()
            indice1 = indice[:samples // 2]
            indice2 = indice[samples // 2:]
            np.random.shuffle(indice2)
            indice = np.concatenate((indice1, indice2))

    for w in range(width):
        temp_prompt = temp_prompts[indice[w]]
        encoded_prompt = model.encode(temp_prompt)
        scores[temp_prompt] = util.dot_score(encoded_prompt, encoded_gt)[0][0].cpu().numpy().tolist()
        scores = search(model, temp_prompt + '<mask>' * 5, gt=gt, scores=scores, width=width, depth=depth - 1, samples=samples)
    return scores


if __name__ == '__main__':
    scoring_model = Model()
    iters = []
    prompts = [
        'รูปของ<mask><mask><mask><mask><mask><mask><mask><mask>',
        'รูปของ<mask><mask><mask><mask><mask><mask><mask><mask>',
        'รูปของ<mask><mask><mask><mask><mask><mask><mask><mask>',
        'รูปของ<mask><mask><mask><mask><mask><mask><mask><mask>',
        'รูปของ<mask><mask><mask><mask><mask><mask><mask><mask>',
    ]
    s = {}
    for i in range(3):
        # explore
        for j in range(5):
            s = search(scoring_model, prompts[j] + "<mask>", depth=3, width=3, index_prop=0.5 if i == 0 else 1, scores=s)
        ak = np.array(list(s.keys()))
        av = torch.tensor(list(s.values()))
        prompts = ak[av.argsort(-1, True)]

        # fine_tune
        for j in range(50):
            if j*50 >= len(prompts):
                break
            prompt = prompts[j*50]  # todo: maybe use j*50 instead
            pts = scoring_model.tokenize(prompt)
            prompt_ids = pts['input_ids'][0]
            for k in range(1, prompt_ids.size(0)):
                s = search(scoring_model, prompt, depth=1, width=3, index_prop=1, scores=s, chosen_index=k)

        ak = np.array(list(s.keys()))
        av = torch.tensor(list(s.values()))
        prompts = ak[av.argsort(-1, True)[:5]]
        print(prompts)
    json.dump(
        s,
        open('out/d.json', 'w')
    )
