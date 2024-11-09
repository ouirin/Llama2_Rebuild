import torch
from model_code.manager import ChatManager
from quant_tool.save_load import load_quant_model

model, tokenizer = load_quant_model('../model_file/simple_quant.pth', '../model_file/sentencepiece.model', torch_dtype=torch.float16)

manager = ChatManager(config=None, model=model, tokenizer=tokenizer, device='cuda')

prompt = "What is the difference between dog and cat?"

for text in manager.generate(prompt, temperature=0.6):

    print(text)
