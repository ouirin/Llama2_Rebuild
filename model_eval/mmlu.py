import os
import torch
import pandas as pd
from tqdm import tqdm
from model_code.manager import ChatManager
from quant_tool.save_load import load_quant_model

# manager = ChatManager.from_pretrained("../model_file", device=torch.device("cpu"))
# tokenizer = manager.tokenizer

model, tokenizer = load_quant_model('../model_file/gptq_quant_32_sample.pth', '../model_file/sentencepiece.model', torch_dtype=torch.float16)

manager = ChatManager(config=None, model=model, tokenizer=tokenizer, device='cuda')

choices = ["A", "B", "C", "D"]
choice_tokens = [tokenizer.encode(choice, bos=False, eos=False)[0] for choice in choices]
char_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

total_correct = 0
total_num = 0


def format_example(df, idx, include_answer=True):

    prompt = df.iloc[idx, 0]
    for i in range(4):
        prompt += "\n{}. {}".format(choices[i], df.iloc[idx, i+1])

    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, 5])

    return prompt


def gen_prompt(df, file):

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(file.split("_")[0])
    k = df.shape[0]
    for i in range(k):
        prompt += format_example(df, i)

    return prompt

for file in os.listdir('test/'):

    df = pd.read_csv('test/' + file, header=None)
    df_five_shot = pd.read_csv('dev/' + file.replace('_test', '_dev'), header=None)
    five_shot_prompt = gen_prompt(df_five_shot, file)
    file_correct = 0
    file_num = df.shape[0]
    dataset = []

    for i in range(df.shape[0]):

        prompt = format_example(df, i, include_answer=False)
        prompt = five_shot_prompt + prompt
        label = df.iloc[i, df.shape[1] - 1]
        label = [char_to_num[char] for char in label]

        logits = manager.generate_logit(prompt)
        torch.cuda.empty_cache()
        # print(f'Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB')
        # print(f'Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2} MB')

        logits = logits[:, choice_tokens]
        preds = logits.argmax(dim=-1)

        file_correct += (preds.cpu() == torch.tensor(label)).sum().item()
        # print(file, file_num, i + 1, file_correct)

    print(file, file_num, i+1, file_correct)

    total_num += file_num
    total_correct += file_correct

print(total_correct / total_num)


# batch generation
# for file in os.listdir('test/'):
#
#     df = pd.read_csv('test/' + file, header=None)
#     df_five_shot = pd.read_csv('dev/' + file.replace('_test', '_dev'), header=None)
#     five_shot_prompt = gen_prompt(df_five_shot, file)
#     dataset = []
#
#     for i in range(df.shape[0]):
#
#         prompt = format_example(df, i, include_answer=False)
#         prompt = [five_shot_prompt + prompt]
#         label = [df.iloc[i, df.shape[1] - 1]]
#         label = [char_to_num[char] for char in label]
#
#         dataset.append({'prompt': prompt, 'label': label})
#
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
#
#     for batch in tqdm(dataloader):
#
#         texts = batch['prompt'][0]
#         queries = [text for text in texts]
#         input_ids = tokenizer(queries, padding='left', max_length=2048).to('cuda')
#
#         outputs = manager.batch_generate(input_ids, max_generated_tokens=128)
#         torch.cuda.empty_cache()
#
#         for idx in range(len(outputs)):
#             output = outputs.tolist()[idx]
#             if tokenizer.pad_id in output:
#                 pad_index = output.index(tokenizer.pad_id)
#                 output = output[:pad_index]
#             response = tokenizer.decode(output)
#             print(response)


# (llama2 don't support batch logit inference)
# for file in os.listdir('test/'):
#
#     df = pd.read_csv('test/' + file, header=None)
#     df_five_shot = pd.read_csv('dev/' + file.replace('_test', '_dev'), header=None)
#     five_shot_prompt = gen_prompt(df_five_shot, file)
#     file_correct = 0
#     file_num = df.shape[0]
#     dataset = []
#
#     for i in range(df.shape[0]):
#
#         prompt = format_example(df, i, include_answer=False)
#         prompt = [five_shot_prompt + prompt]
#         label = [df.iloc[i, df.shape[1] - 1]]
#         label = [char_to_num[char] for char in label]
#
#         dataset.append({'prompt': prompt, 'label': label})
#
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
#
#     for batch in tqdm(dataloader):
#
#         texts = batch['prompt'][0]
#         queries = [text for text in texts]
#         input_ids = tokenizer(queries, padding='left', max_length=2048).to('cuda')
#
#         logits = manager.batch_generate_logit(input_ids, max_generated_tokens=128)
#         torch.cuda.empty_cache()
#
#         logits = logits[:, choice_tokens]
#         preds = logits.argmax(dim=-1)
#
#         file_correct += (preds.cpu() == torch.tensor(batch['label'][0])).sum().item()
#         print(file, file_num, file_correct)
#
#     print(file, file_num, file_correct)
#
#     total_num += file_num
#     total_correct += file_correct
#
# print(total_correct / total_num)
#








