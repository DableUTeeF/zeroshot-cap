from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
import os
import pandas as pd
import tqdm


def gen_mtl():
    tr_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to('cuda')
    eng_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")
    tha_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tha_Thai")

    src = '/home/palm/data/coco/annotations/annotations'
    # for idx, (file, name) in enumerate(zip(
    #     ['caption_human_thai_val2017.json', 'caption_human_thai_train2017.json'],
    #     ['val', 'train']
    # )):
    for idx, (file, name) in enumerate(zip(
            ['caption_human_thai_train2017.json'],
            ['train']
    )):
        dataset = annDataset(src, file)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        with open(f'data/{name}.csv', 'w') as wr:
            wr.write('gt,en,th\n')
        for tha_caption in tqdm.tqdm(dataloader):
            inputs = tha_tokenizer(tha_caption, return_tensors="pt", padding=True).to('cuda')
            translated_tokens = tr_model.generate(
                **inputs,
                forced_bos_token_id=tha_tokenizer.lang_code_to_id["eng_Latn"],
            )
            eng_caption = tha_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            inputs = eng_tokenizer(eng_caption, return_tensors="pt", padding=True).to('cuda')
            translated_tokens = tr_model.generate(
                **inputs,
                forced_bos_token_id=eng_tokenizer.lang_code_to_id["tha_Thai"],
            )
            translated_caption = eng_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            df = {'gt': [], 'en': [], 'th': []}
            for i in range(len(tha_caption)):
                eng_cap = eng_caption[i]
                tha_cap = tha_caption[i]
                mtl_cap = translated_caption[i]
                df['gt'].append(tha_cap)
                df['en'].append(eng_cap)
                df['th'].append(mtl_cap)
            df = pd.DataFrame(df)
            df.to_csv(f'data/{name}.csv', mode='a', header=False, index=False)


class annDataset(Dataset):
    def __init__(self, src, file):
        self.data = json.load(open(os.path.join(src, file)))['annotations']['caption_thai']

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class MTLDataset(Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv, header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        return data[0], data[2]


if __name__ == '__main__':
    gen_mtl()
