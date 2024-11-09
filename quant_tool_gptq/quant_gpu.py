import torch
from model_code.loader import load_model_and_tokenizer
from quant_tool_gptq.save_load import save_quant_model
from quant_tool_gptq.quant_util import load_calibration, bind_quantizer, GptqQuantizer, cover_layer

# load model
config, model, tokenizer = load_model_and_tokenizer("../model_file")
print(model)

# load data
data = load_calibration(tokenizer, calibration_file="ppl_dataset", n_sample=32, sample_length=2048)

# generate block_input
prepared_input = []
current_h = []
with torch.no_grad():
    for batch in data:
        temp = model.prepare_input(batch, all_kv_cache=None)
        prepared_input.append(temp)
        current_h.append(temp[0])

# for each block, move to gpu、bind_quantizer、forward、get_quantized_linear、move to cpu
for layer_idx in range(32):

    print(layer_idx)
    layer = model.layers[layer_idx]
    layer.to("cuda")

    qlayers = bind_quantizer(layer, linear_bit=4, linear_group=32)

    next_h = tuple()
    for h, (_, attention_mask, rotary_complex) in zip(current_h, prepared_input):
        with torch.no_grad():
            h, _ = layer(x=h.to("cuda"), rotary_complex=rotary_complex.to("cuda"), attention_mask=attention_mask.to("cuda"), kv_cache=None)
            next_h += (h,)
    current_h = next_h

    for path, module in qlayers.items():
        cover_layer(layer, path, module.get_quantized_linear())
        print(f"{path} - finished")

    layer.to("cpu")
    torch.cuda.empty_cache()

# quant lm_head
model.final_ln.to("cuda")
model.lm_head.to("cuda")
quantizer = GptqQuantizer(model.lm_head, bit=4, group_size=32)
with torch.no_grad():
    for h in current_h:
        model.lm_head(model.final_ln(h))
setattr(model, 'lm_head', quantizer.get_quantized_linear())

# save model
model.to("cpu")
print(model)
save_quant_model(model, "../model_file/gptq_quant_32_sample.pth")


