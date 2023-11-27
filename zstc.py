import os

import torch
import torch.nn as nn

from transformers import VisionEncoderDecoderModel, AutoModelForSeq2SeqLM, CLIPModel, AutoTokenizer, CLIPProcessor, GPT2TokenizerFast, ViTFeatureExtractor, GPT2LMHeadModel
from PIL import Image
import json
import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


class ZeroShotThaiCapgen(nn.Module):

    def __init__(self):
        super().__init__()
        self.ic_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.ic_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.ic_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        # self.tr_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        # self.tr_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")
        self.tr_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tr_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.tr_tokenizer.src_lang = "en"

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.ppl_model = GPT2LMHeadModel.from_pretrained("flax-community/gpt2-base-thai")
        self.ppl_tokenizer = GPT2TokenizerFast.from_pretrained("flax-community/gpt2-base-thai")
        self.ppl_tokenizer.pad_token_id = self.ppl_tokenizer.eos_token_id
        print('init')
        # self.ic_model.eval().cuda()
        # self.tr_model.eval().cuda()
        # self.clip_model.eval().cuda()
        # self.ppl_model.eval().cuda()
        # print('cuda')

    def img2eng(
            self,
            pil_image,
            gen_kwargs,
    ):

        pixel_values = self.ic_feature_extractor(images=[pil_image], return_tensors="pt").pixel_values.to('cuda')

        output_ids = self.ic_model.generate(pixel_values, **gen_kwargs)

        eng_caption = self.ic_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return eng_caption

    def select_caption(
            self,
            pil_image,
            captions,
            use_ppl=False,
    ):
        inputs = self.clip_processor(images=pil_image, text=captions, padding=True, return_tensors="pt", max_length=77, truncation=True).to('cuda')

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(inputs['pixel_values'], output_hidden_states=True)
            text_features = self.clip_model.get_text_features(inputs['input_ids'])

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            score = (text_features @ image_features.T).view(-1)
            # print(score)

        if use_ppl:
            for i in range(len(captions)):
                s = self.compute_ppl(captions[i]) / 100.0
                # print(s, score[i])
                score[i] -= s

        # print(score)
        idx = score.argmax()

        return captions[idx]

    def eng2tha(
            self,
            eng_caption,
            gen_kwargs,
    ):
        inputs = self.tr_tokenizer(eng_caption, return_tensors="pt").to('cuda')
        translated_tokens = self.tr_model.generate(
            **inputs,
            forced_bos_token_id=self.tr_tokenizer.get_lang_id("th"),
            **gen_kwargs
        )
        tha_caption = self.tr_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        return tha_caption

    @torch.no_grad()
    def forward(
            self,
            pil_image,
            num_beams=5,
            max_length=30,
            num_return_sequences=5,
    ):

        gen_kwargs = {
            "num_beams": num_beams,
            "max_length": max_length,
            "num_return_sequences": num_return_sequences
        }

        eng_captions = self.img2eng(pil_image, gen_kwargs)
        # print(eng_captions)

        selected_eng_caption = self.select_caption(pil_image, eng_captions, use_ppl=False)

        tha_captions = self.eng2tha(selected_eng_caption, gen_kwargs)
        # print(tha_captions)

        selected_tha_caption = self.select_caption(pil_image, tha_captions, use_ppl=True)

        return selected_tha_caption, selected_eng_caption

    def compute_ppl(
            self,
            text,
    ):
        encodings = self.ppl_tokenizer(
            text,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=30,
            return_tensors="pt",
            return_attention_mask=True,
        ).to('cuda')

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        with torch.no_grad():
            out_logits = self.ppl_model(encoded_texts, attention_mask=attn_masks).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = encoded_texts[..., 1:].contiguous()
        shift_attention_mask_batch = attn_masks[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        perplexity = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        return perplexity[0]  # one text only


if __name__ == '__main__':
    capgen = ZeroShotThaiCapgen().eval().cuda()
    print('cuda')
    src = '/home/palm/data/coco/images'
    # outputs = json.load(open('outputs.json'))
    # df = pd.read_csv('outputs.csv', header=None)
    # finished = list(outputs.keys()) + list(df[0])
    print('starting')
    with open('outputs2.csv', 'a') as wr:
        for folder in ['val2017']:
            print(folder)
            for file in os.listdir(os.path.join(src, folder)):
                # if file in outputs:
                #     continue
                image = Image.open(os.path.join(src, folder, file)).convert('RGB')
                outputs[file] = capgen(image)
                wr.write(f'{file},"{outputs[file][0]}","{outputs[file][1]}"\n')
    # json.dump(outputs,
    #           open('outputs2.json', 'w'))
