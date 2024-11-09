import torch
from model_code.loader import load_model_and_tokenizer


def top_p_sampling(logits, top_p=0.8, temperature=1.0):

    probs = torch.softmax(logits / temperature, dim=-1)
    probs_sort, indices = torch.sort(probs, dim=-1, descending=True)

    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_p] = 0.0
    probs_sort = probs_sort / torch.sum(probs, dim=-1, keepdim=True)

    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(indices, -1, next_token)

    return next_token


class ChatManager:

    def __init__(self, config, model, tokenizer, device=None):

        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        self.device = device
        self.model.to(self.device)

    @staticmethod
    def from_pretrained(path=None, device=None):

        config, model, tokenizer = load_model_and_tokenizer(path)

        model.to(device=device)

        return ChatManager(config, model, tokenizer, device=device)

    def generate(self, prefix_text, max_generated_tokens=150, top_p=0.9, temperature=0.6):

        prefix_ids = self.tokenizer.encode(prefix_text, bos=True, eos=False)

        input_ids = torch.LongTensor([prefix_ids])

        all_kv_cache = None
        generated_tokens = []

        while len(generated_tokens) < max_generated_tokens:

            with torch.no_grad():

                loss, logits, all_kv_cache = self.model(input_ids=input_ids.to(self.device), all_kv_cache=all_kv_cache)

                if temperature > 0:
                    next_token = top_p_sampling(logits[0, -1], top_p, temperature).item()
                else:
                    next_token = torch.argmax(logits[0, -1], dim=-1).item()

            generated_tokens += [next_token]

            if next_token == self.tokenizer.eos_id:
                break

            response_text = self.tokenizer.decode(generated_tokens)

            if response_text and response_text[-1] != " ":
                yield response_text

            input_ids = torch.tensor([[next_token]]).long()

        return self.tokenizer.decode(generated_tokens)

    def generate_logit(self, prefix_text):

        prefix_ids = self.tokenizer.encode(prefix_text, bos=True, eos=False, truncation=2048)

        input_ids = torch.LongTensor([prefix_ids])

        all_kv_cache = None

        with torch.no_grad():

            self.model.to("cuda")

            _, logits, _ = self.model(input_ids=input_ids.to(self.device), all_kv_cache=all_kv_cache)

            self.model.to("cpu")   # for gpu memory release

        return logits[:, -1]

    def batch_generate(self, input_ids, max_generated_tokens=512, top_p=0.9, temperature=0.6):

        all_kv_cache = None
        generated_tokens = [[] for _ in range(input_ids.shape[0])]
        finished = [False] * input_ids.shape[0]
        all_seq = []

        for i in range(input_ids.shape[0]):
            if self.tokenizer.pad_id in input_ids[i]:
                pad_index = (input_ids[i] == self.tokenizer.pad_id).int().argmax().item()
                all_seq.append(pad_index)
            else:
                all_seq.append(len(input_ids[i]))

        min_seq = min(all_seq)
        original_input_ids = input_ids
        input_ids = input_ids[:, :min_seq]

        for _ in range(max_generated_tokens):

            with torch.no_grad():

                loss, logits, all_kv_cache = self.model(input_ids=input_ids.to(self.device), all_kv_cache=all_kv_cache)

                cur_pos = all_kv_cache[0][0].shape[1]

                next_tokens = []

                for i in range(input_ids.shape[0]):

                    if cur_pos < original_input_ids[i].shape[0]:  # if cur_pos has token, use the original one
                        if original_input_ids[i][cur_pos] != self.tokenizer.pad_id:
                            next_token = original_input_ids[i][cur_pos].item()
                            next_tokens.append(next_token)
                        else:
                            next_token = top_p_sampling(logits[i, -1], top_p, temperature).item()
                            next_tokens.append(next_token)
                    else:
                        next_token = top_p_sampling(logits[i, -1], top_p, temperature).item()
                        next_tokens.append(next_token)

                    if cur_pos < original_input_ids[i].shape[0]:
                        if original_input_ids[i][cur_pos] != self.tokenizer.pad_id:
                            pass
                        elif finished[i] is False:
                            generated_tokens[i] += [next_token]
                        else:
                            pass
                    elif finished[i] is False:
                        generated_tokens[i] += [next_token]
                    else:
                        pass

                    if next_token == self.tokenizer.eos_id:
                        finished[i] = True

                if all(finished):
                    break

                input_ids = torch.tensor(next_tokens).unsqueeze(1).long()

        max_length = max(len(generated_token) for generated_token in generated_tokens)
        generated_tokens = [generated_token + [self.tokenizer.pad_id] * (max_length - len(generated_token)) for generated_token in generated_tokens]
        generated_tokens = torch.tensor(generated_tokens)

        return generated_tokens

    # sequence with pad_id don't work
    def batch_generate_logit(self, input_ids):

        with torch.no_grad():

            _, logits, _ = self.model(input_ids=input_ids.to(self.device), all_kv_cache=None)

        return logits[:, -1]
