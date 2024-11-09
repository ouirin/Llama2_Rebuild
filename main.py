import torch
from model_code.manager import ChatManager

manager = ChatManager.from_pretrained("model_file", device=torch.device("cpu"))

prompt = "What is the difference between dog and cat?"  

for text in manager.generate(prompt, temperature=0):

    print(text)


# ~~~use original weight
# Dogs are more loyal than cats.
# Dogs are more loyal than cats. They are more likely to be loyal to their owners. They are also more likely to be loyal to their owners. They are also more likely to be loyal to their owners. They are also more likely to be loyal to their owners. They are also more likely


# ~~~use hugging face weight
from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("../llama2")
model = LlamaForCausalLM.from_pretrained("../llama2")
# ids = tokenizer.encode("What is the difference between dog and cat?")
# input_ids = torch.LongTensor([ids])
# out = model.generate(input_ids=input_ids, max_length=150, do_sample=False, temperature=0)
# out_text = tokenizer.decode(out[0])
# Dogs are more loyal than cats.
# Dogs are more loyal than cats. They are more likely to be loyal to their owners. They are also more likely to be loyal to their owners. They are also more likely to be loyal to their owners. They are also more likely to be loyal to their owners. They are also more likely


# ~~~use hugging face weight with my code
# manager = ChatManager.from_pretrained("model_file", device=torch.device("cpu"))
# Dogs are more loyal than cats. Dog is a domesticated animal and cat is a wild animal.
# Dogs are more loyal than cats.
