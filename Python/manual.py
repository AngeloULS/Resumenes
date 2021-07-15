# import the necessary

import torch
from transformers import BertForMaskedLM, BertTokenizer

# create the tokenizer and the model

tokenizer = BertTokenizer.from_pretrained("C:\\Users\\angel\\Documents\\ULS 2021\\IA\\Repositorio\\Modelo_Beto\\", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("C:\\Users\\angel\\Documents\\ULS 2021\\IA\\Repositorio\\Modelo_Beto\\")
e = model.eval()

# Now test it

text = "[CLS] Para solucionar los [MASK] de Chile, el presidente debe [MASK] de inmediato. [SEP]"
masked_indxs = (4,11)

tokens = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
tokens_tensor = torch.tensor([indexed_tokens])

predictions = model(tokens_tensor)[0]

for i,midx in enumerate(masked_indxs):
    idxs = torch.argsort(predictions[0,midx], descending=True)
    predicted_token = tokenizer.convert_ids_to_tokens(idxs[:5])
    print('MASK',i,':',predicted_token)
