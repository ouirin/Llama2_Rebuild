import torch
from quant_tool.quant_linear import QuantLinear
from model_code import model as modeling
from model_code.model import Llama2Model, Llama2Config
from model_code.tokenizer import Llama2Tokenizer


def save_quant_model(q_model, path):

    torch.save(q_model.state_dict(), path)


def load_quant_model(model_path, tokenizer_path, torch_dtype):

    modeling.Linear = QuantLinear
    model = Llama2Model(Llama2Config(), torch_dtype)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    tokenizer = Llama2Tokenizer(tokenizer_path)

    return model, tokenizer