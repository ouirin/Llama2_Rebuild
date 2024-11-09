from model_code.loader import load_model_and_tokenizer
from quant_tool_gptq.quant_util import quant_model
from quant_tool_gptq.save_load import save_quant_model

config, model, tokenizer = load_model_and_tokenizer("../model_file")
print(model)

model_q = quant_model(model=model, tokenizer=tokenizer)
print(model_q)

save_quant_model(model_q, "../model_file/gptq_quant_32_sample.pth")
