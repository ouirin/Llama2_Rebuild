import torch
import math
import torch.nn.functional as F
from datasets import load_dataset
from model_code.manager import ChatManager
from quant_tool.save_load import load_quant_model

# traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir='ppl_dataset')
# testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir='ppl_dataset')

testdata = load_dataset("ppl_dataset", split='test')
testdata = "\n\n".join(testdata['text'])
print(len(testdata))

model, tokenizer = load_quant_model('../model_file/gptq_quant_32_sample.pth', '../model_file/sentencepiece.model', torch_dtype=torch.float16)
manager = ChatManager(config=None, model=model, tokenizer=tokenizer, device='cuda')

input_ids = tokenizer.encode(testdata)
input_ids = torch.LongTensor([input_ids])

num_sample = input_ids.numel() // 2048
print(num_sample)
losses = []

with torch.no_grad():

    for i in range(num_sample):

        print(i)
        input_id = input_ids[:, (i * 2048):((i + 1) * 2048)].to('cuda')

        _, logits, all_kv_cache = model(input_ids=input_id, all_kv_cache=None)

        n_classes = tokenizer.n_words
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = input_id[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, n_classes), shift_labels.view(-1))

        losses.append(loss)

        del all_kv_cache
        torch.cuda.empty_cache()
        # print(f'Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB')
        # print(f'Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2} MB')

        avg = sum(losses) / len(losses)
        print(f'ppl:{math.exp(avg):.6f}')

avg = sum(losses) / len(losses)
print(f'ppl:{math.exp(avg):.6f}')

