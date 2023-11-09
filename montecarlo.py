import torch
import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch import nn
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('xlm-roberta-large')
        self.LinearTransformation = torch.nn.Linear(1024, 640)
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        self.mask_token_id = self.tokenizer.mask_token_id

    def forward(self, *args, **kwargs):
        embs = self.transformer(*args, **kwargs)[0]
        att = kwargs['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)

    def encode(self, text):
        return self.tokenizer(text, return_tensors='pt')

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids)


def search(model,
           prompt='รูปของ<mask><mask><mask><mask><mask><mask><mask><mask>',
           gt='image of a sleeping girl amidst elegant swirling water',
           depth=10,
           width=20,
           ):
    scores = {}
    prompt_ids = model.encode(prompt)['input_ids'][0]
    for d in range(depth):
        for w in range(width):
            chosen_idx = torch.where(prompt_ids == model.mask_token_id)[0].numpy()
            idx = np.random.choice(chosen_idx)
            pass
    return


if __name__ == '__main__':
    model = Model()
    d = '/home/palm/.cache/huggingface/hub/models--M-CLIP--XLM-Roberta-Large-Vit-B-16Plus/blobs/a7124266439dd6bce544c23b57249a2ca764dbcd38e8eab16ce272c28c27b049'
    cp = torch.load(d)
    model.load_state_dict(cp)
    search(model)
