import torch
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EncoderDecoderModel,
    BertConfig, 
    EncoderDecoderConfig, 
    BertTokenizer,
    BertTokenizerFast,
)

from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig

#### Utilizamos GPU para el entrenamiento ####
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(device)

#### Cargamos el modelo BETO para Realizar Fine-Tuning seq2seq ####
# BETO: https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/tree/main
# DATASET: https://huggingface.co/datasets/viewer/?dataset=mlsum

#RUTA_MODELO = "C:\\Users\\angel\\Documents\\ULS 2021\IA\\Repositorio\\Resumenes\\Modelo_Beto\\"
RUTA_MODELO = 'D:\\Fine-Tuning\\Modelo_Beto'
MODELO_SAVE = "C:\\Users\\angel\\Documents\\ULS 2021\IA\\Repositorio\\Resumenes\\Modelo_Beto\\Modelo"
archivos = {
    "train":"data/es_train.jsonl",
    "validation":"data/es_val.jsonl",
    "test":"data/es_test.jsonl"
}

raw_datasets = load_dataset("json",data_files=archivos, cache_dir="cache")

#### Configuracion modelo Encoder y Decoder ####
# CONFIG: https://huggingface.co/transformers/model_doc/encoderdecoder.html 
config_encoder = BertConfig()
config_decoder = BertConfig()

config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

# Initializing a Bert2Bert model from the bert-base-uncased style configurations
model = EncoderDecoderModel(config=config)

# Accessing the model configuration
config_encoder = model.config.encoder
config_decoder  = model.config.decoder

# set decoder config to causal lm
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True

# Saving the model, including its configuration
model.save_pretrained(MODELO_SAVE)

#decoder_start_token_id
tokenizer = BertTokenizerFast.from_pretrained(RUTA_MODELO)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
print(tokenizer.bos_token_id)



encoder_decoder_config = EncoderDecoderConfig.from_pretrained(RUTA_MODELO)
model = EncoderDecoderModel.from_pretrained(RUTA_MODELO, config=encoder_decoder_config)

###decoder_start_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

##
model.encoder.resize_token_embeddings(len(tokenizer))
model.decoder.resize_token_embeddings(len(tokenizer))

if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

def preprocess_function(examples):
    inputs = examples["text"]
    targets = examples["summary"]

    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=4)
    outputs = tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=4)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, padding="max_length", truncation=True)
                                                     
    examples["input_ids"] = inputs.input_ids                                                               
    examples["attention_mask"] = inputs.attention_mask                                                     
    examples["decoder_input_ids"] = outputs.input_ids                                                      
    examples["labels"] = outputs.input_ids.copy() 

    # mask loss for padding                                                                             
    examples["labels"] = [                                                                                 
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in examples["labels"]
    ]                     
    examples["decoder_attention_mask"] = outputs.attention_mask 

    return examples

if "train" not in raw_datasets:
    raise ValueError("--do_train requires a train dataset")

#### TRAIN ####

column_train = raw_datasets["train"].column_names
train_dataset = raw_datasets["train"]

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    #num_proc=None,
    remove_columns=column_train,
    #load_from_cache_file=not False,
    desc="Running tokenizer on train dataset",
)

#### VALIDATION ####

colum_eval = raw_datasets["validation"].column_names
eval_dataset = raw_datasets["validation"]

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    #num_proc=data_args.preprocessing_num_workers,
    remove_columns=colum_eval,
    #load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on eval dataset",
)

# Data collator
label_pad_token_id = -100 if True else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 ,
)

# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    #args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics if training_args.predict_with_generate else None,
)

# Training

train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
max_train_samples = (
    len(train_dataset)
)

metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()