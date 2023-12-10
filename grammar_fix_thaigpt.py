import pandas as pd
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import time
import json
import os


def generate(input, instruction):
    prompt = template['prompt_input'].format(
        instruction=instruction,
        input=input
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output


if __name__ == '__main__':
    device = 'cpu'
    prompt_template = ""
    server_name = "0.0.0.0"
    base_model = 'daryl149/llama-2-7b-chat-hf'
    lora_weights = 'openthaigpt/openthaigpt-1.0.0-alpha-7b-chat'
    load_8bit = False
    temperature = 0.1
    top_p = 0.75
    top_k = 40
    num_beams = 4
    max_new_tokens = 128

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    model = torch.compile(model)

    template = {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:"
    }
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )
    
    if os.path.exists('/home/palm/data/coco/annotations/'):
        val = json.load(open('/home/palm/data/coco/annotations/annotations/caption_human_thai_val2017.json'))
        length = 581
    else:
        val = json.load(open('/media/palm/data/coco/annotations/caption_human_thai_val2017.json'))
        length = 3557
    # images = [int(x['image_id']) for x in val['annotations']]
    gt = {}
    for ann in val['annotations']:
        if int(ann['image_id']) not in gt:
            gt[int(ann['image_id'])] = []
        text = 1
        gt[int(ann['image_id'])].append(text)
    # instruction = 'แก้ประโยคต่อไปนี้ให้มีไวยากรณ์ที่ถูกต้อง'
    # input = 'สองหมาอยู่กำลังวิ่งบนหนึ่งสนาม'
    df = pd.read_csv('outputs.csv', header=None)
    df['grammared'] = ''
    df['tled'] = ''
    df['time'] = 0
    for i, row in df.iterrows():
        if i < length:
            continue
        if int(row[0][:-4]) not in gt or len(gt[int(row[0][:-4])]) < 2:
            continue
        t = time.time()
        df['grammared'].iloc[i] = generate(row[1], 'แก้ประโยคต่อไปนี้ให้มีไวยากรณ์ที่ถูกต้อง').split('\n')[-1].replace('</s>', '')
        df['tled'].iloc[i] = generate(row[2], 'แปลประโยคต่อไปนี้จากภาษาอังกฤษเป็นภาษาไทยให้ถูกต้อง').split('\n')[-1].replace('</s>', '')
        df['time'].iloc[i] = time.time() - t
        df.to_csv('llama_output.csv', index=False)
