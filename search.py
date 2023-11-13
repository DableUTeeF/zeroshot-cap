import numpy as np


def search(model,
           prompt,
           gt='image of a sleeping girl amidst elegant swirling water',
           depth=10,
           width=20,
           scores=None,
           ):
    if depth < 1:
        return scores
    if scores is None:
        scores = {prompt: get_score(model, prompt, gt)}
    for d in range(depth):
        input_ids = model.tokenize(prompt)
        for w in range(width):
            temp_input_ids = random_inputs_ids(input_ids)
            temp_prompt = model.detokenize(temp_input_ids)
            if temp_prompt not in scores:
                scores[temp_prompt] = get_score(model, prompt, gt)
            scores = search(model, temp_prompt, scores=scores, depth=depth-1)
    return scores
