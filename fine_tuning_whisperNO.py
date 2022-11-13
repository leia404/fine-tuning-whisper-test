import argparse
import collections
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import pandas as pd
import torch
# if not hasattr(collections, "Container"):
#     import collections.abc
#     collections.Container = collections.abc.Container
import transformers
import wandb
from datasets import Audio, ClassLabel, Dataset, load_dataset, load_metric
from IPython.display import HTML, display
from pydub import AudioSegment
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperFeatureExtractor, WhisperProcessor,
                          WhisperTokenizer)

wandb.init(project="fine-tuning-whisperNO", entity="janinerugayan")

# https://huggingface.co/transformers/main_classes/logging.html
# verbosity set to print errors only, by default it is set to 30 = error and warnings
transformers.logging.set_verbosity(40)


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\*]'
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch


def prepare_dataset(batch):
    audio = batch["audio"]
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def load_dataset_from_files(data_dir_list:list[str], csv_export_dir:str, split_ratio=0.1, csv_export=True):
    frames = []
    for path in data_dir_list:
        source = os.path.basename(os.path.dirname(path))
        wavfile_data = []
        textfile_data = []
        for (root, dirs, files) in os.walk(path, topdown=True):
            if source == "Rundkast":  # to modify depending on Rundkast cuts folder name
                for fn in files:
                    if fn.endswith(".wav"):
                        wav_id = os.path.splitext(fn)[0]
                        path = os.path.join(root, fn)
                        wavfile_data.append((wav_id, fn, path, source))
                    elif fn.endswith(".txt"):
                        text_id = os.path.splitext(fn)[0]
                        with open(os.path.join(root, fn), encoding="utf-8") as text_file:
                            text = text_file.read()
                        textfile_data.append((text_id, text))
            else:
                for fn in files:
                    if fn.endswith(".wav"):
                        wav_id = os.path.splitext(fn)[0]
                        path = os.path.join(root, fn)
                        wavfile_data.append((wav_id, fn, path, source))
                    elif fn.endswith(".txt-utf8"):
                        text_id = os.path.splitext(fn)[0]
                        with open(os.path.join(root, fn), encoding="utf-8-sig") as text_file:
                            text = text_file.read()
                        textfile_data.append((text_id, text))
        df_wav = pd.DataFrame(wavfile_data, columns=["segment_id", "wav_file", "path", "source"])
        df_wav = df_wav.set_index("segment_id")
        df_text = pd.DataFrame(textfile_data, columns=["segment_id", "text"])
        df_text = df_text.set_index("segment_id")
        dataset_df = df_wav.merge(df_text, left_index=True, right_index=True)
        frames.append(dataset_df)
    # concat to full dataframe and convert to Dataset with special characters removed
    full_dataset_df = pd.concat(frames)
    raw_dataset = Dataset.from_pandas(full_dataset_df)
    # raw_dataset = raw_dataset.map(remove_special_characters)
    # split dataset
    raw_dataset = raw_dataset.train_test_split(test_size=split_ratio)
    # save copy of dataset
    if csv_export == True:
        df_train = pd.DataFrame(raw_dataset["train"])
        df_train.to_csv(os.path.join(csv_export_dir, "train_set.csv"))
        df_dev = pd.DataFrame(raw_dataset["test"])
        df_dev.to_csv(os.path.join(csv_export_dir, "dev_set.csv"))
    # loading audio
    dataset = raw_dataset.cast_column("path", Audio())
    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    # preprocess dataset
    dataset = dataset.map(prepare_dataset,
                          remove_columns=dataset.column_names["train"],
                          num_proc=4)
    return raw_dataset, dataset




# ---------------------------------------------------
# LOAD PRETRAINED MODEL
# ---------------------------------------------------

parser=argparse.ArgumentParser()
parser.add_argument("--original_model",         type=str)
parser.add_argument("--fine_tuned_model_ver",   type=str)
parser.add_argument("--export_model_dir",       type=str)
parser.add_argument("--num_train_epochs",       type=int)
parser.add_argument("--learning_rate",          type=float)
parser.add_argument("--used_asd_metric",        type=int)
args = parser.parse_args()




# ---------------------------------------------------
# LOAD PRETRAINED MODEL
# ---------------------------------------------------

print("Loading pretrained model " + args.original_model)

model_name = args.original_model

# load feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

# load tokenizer
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Norwegian", task="transcribe")

# create processor
processor = WhisperProcessor.from_pretrained(model_name, language="Norwegian", task="transcribe")

# load pretrained model
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []  # no tokens are suppressed




# ---------------------------------------------------
# LOAD DATASET FROM CSV FILES
# ---------------------------------------------------

print("Loading dataset direct from data dir to pandas dataframe")

# data_dir_list = ["../../datasets/NordTrans_TUL/train/Stortinget/",
#                  "../../datasets/NordTrans_TUL/train/NRK/",
#                  "../../datasets/NordTrans_TUL/train/Rundkast_cuts_random25per_30secmax/"]

data_dir_list = ["../../datasets/NordTrans_TUL/train_small/Stortinget/", 
                    "../../datasets/NordTrans_TUL/train_small/NRK/", 
                    "../../datasets/NordTrans_TUL/train_small/Rundkast/"]

csv_export_dir = "../../model_ckpts/" + args.fine_tuned_model_ver + "/runs/"

raw_dataset, dataset = load_dataset_from_files(data_dir_list, csv_export_dir, split_ratio=0.1, csv_export=True)

print(raw_dataset)
print(dataset)




# ---------------------------------------------------
# SET-UP TRAINER
# ---------------------------------------------------

print("Setting up the trainer")

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# initialize the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# evaluation metrics
if args.use_asd_metric == 1:
    # https://huggingface.co/transformers/main_classes/logging.html
    # verbosity set to print errors only, by default it is set to 30 = error and warnings
    transformers.logging.set_verbosity(40)
    # The bare Bert Model transformer outputting raw hidden-states without any specific head on top.
    metric_modelname = 'ltgoslo/norbert'
    metric_model = BertModel.from_pretrained(metric_modelname)
    metric_tokenizer = AutoTokenizer.from_pretrained(metric_modelname)

    asd_metric = load_metric("asd_metric.py")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        asd = asd_metric.compute(model=metric_model, tokenizer=metric_tokenizer, reference=label_str, hypothesis=pred_str)
        return {"asd": asd}

elif args.use_asd_metric == 0:
    wer_metric = load_metric("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str) 
        return {"wer": wer}


repo_local_dir = "../../model_ckpts/" + args.fine_tuned_model_ver + "/"

# training arguments for whisper
training_args = Seq2SeqTrainingArguments(
    output_dir=repo_local_dir,  
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=args.learning_rate,
    warmup_steps=500,
    num_train_epochs=args.num_train_epochs,
    # max_steps=4000,  # original from tutorial
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    report_to="wandb",
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=1,
    save_total_limit=2
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)




# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------

finetuned_model_dir = args.export_model_dir
log_dir = "../../model_ckpts/" + args.fine_tuned_model_ver + "/runs/"

torch.cuda.empty_cache()
print("Training starts")
trainer.train()
# trainer.train("../../model_ckpts/fine-tuning_wav2vec2_v17/checkpoint-15000")

log_history_fn = os.path.join(log_dir, "log_history.txt")
with open(log_history_fn, "w") as f:
    f.write(json.dumps(trainer.state.log_history))

print("Saving fine-tuned model")
model.save_pretrained(save_directory=finetuned_model_dir)
processor.save_pretrained(save_directory=finetuned_model_dir)
