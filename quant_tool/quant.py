from model_code.loader import load_model_and_tokenizer
from quant_tool.quant_util import quant_model
from quant_tool.save_load import save_quant_model

config, model, tokenizer = load_model_and_tokenizer("../model_file")
print(model)

model_q = quant_model(model=model)
print(model_q)

save_quant_model(model_q, "../model_file/simple_quant.pth")
