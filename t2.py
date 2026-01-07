import re
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader

class SentencePieceTokenizer:
    def __init__(self, model_path='./sp.model'):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)
    
tokenizer = SentencePieceTokenizer()


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)                                #A
        print(f'共有 {len(token_ids)} 个词汇')

        for i in range(0, len(token_ids) - max_length, stride):          #B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):                                                     #C
        return len(self.input_ids)

    def __getitem__(self, idx):                                            #D
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)       #B
    dataloader = DataLoader(
          dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          drop_last=drop_last,                                        #C
          num_workers=0                                               #D
    )

    return dataloader


RAW_TEXT_PATH = 'ddia_c0.txt'

# spm.SentencePieceTrainer.Train(
#     input=RAW_TEXT_PATH,
#     model_prefix="sp",
#     vocab_size=2914,
#     model_type="unigram",
#     character_coverage=0.9995,
#     byte_fallback=True
# )


with open(RAW_TEXT_PATH, 'r') as f:
    raw_text = f.read()

# dataloader = create_dataloader_v1(
#       raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
# data_iter = iter(dataloader)              
# first_batch = next(data_iter)
# print(first_batch)
# second_batch = next(data_iter)
# print(second_batch)

max_length = 4
dataloader = create_dataloader_v1(
      raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

vocab_size = 10042
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

attn_scores = torch.empty(8, 8)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
