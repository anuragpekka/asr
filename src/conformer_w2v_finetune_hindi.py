import torch
import torchaudio
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    Wav2Vec2Processor,
    Wav2Vec2ConformerForCTC
)
from dataclasses import dataclass
import numpy as np
from evaluate import load 

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
# Load processor (ensure you have the correct model name)
model = Wav2Vec2ConformerForCTC.from_pretrained(
    "facebook/wav2vec2-conformer-rope-large-960h-ft",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id  # Critical for alignment
)

# Load your audio file (ensure it's sampled at 16kHz)
waveform, sample_rate = torchaudio.load("common_voice_hi_23795238.wav")

# If your audio is not at 16kHz, resample it
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Process the audio
inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

# Perform inference
with torch.no_grad():
    logits = model(inputs.input_values).logits

# Decode the predicted IDs to text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print("Transcription:", transcription[0])

# Load different slices of the train split
train_dataset = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="train[:2%]", trust_remote_code=True)
validation_dataset = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="train[:1%]", trust_remote_code=True)
test_dataset = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="train[:1%]", trust_remote_code=True)

# Function to load and preprocess audio (creates 'speech' column)
def preprocess_audio(batch):
    speech_array, sample_rate = torchaudio.load(batch["path"])
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        speech_array = resampler(speech_array)
    batch["speech"] = speech_array.squeeze().numpy()
    batch["sampling_rate"] = 16000
    batch["text"] = batch["sentence"].lower()  # Normalize text
    return batch

# Apply audio preprocessing FIRST
train_dataset = train_dataset.map(preprocess_audio)
validation_dataset = validation_dataset.map(preprocess_audio)
test_dataset = test_dataset.map(preprocess_audio)

# Function to prepare input_values and labels (requires 'speech' column)
def prepare_dataset(batch):
    input_values = processor(
        batch["speech"], 
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values.squeeze().numpy()
    
    with processor.as_target_processor():
        labels = processor.tokenizer(batch["text"]).input_ids
    
    batch["input_values"] = input_values
    batch["labels"] = labels
    return batch

# Apply dataset preparation SECOND
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
validation_dataset = validation_dataset.map(prepare_dataset, remove_columns=validation_dataset.column_names)

@dataclass
class CustomCTCCollator:
    processor: any
    padding: bool = True

    def __call__(self, batch):
        # Extract input_values and labels
        input_values = [{"input_values": item["input_values"]} for item in batch]
        labels = [{"input_ids": item["labels"]} for item in batch]

        # Pad input_values
        input_batch = self.processor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels and replace padding with -100
        labels_batch = self.processor.tokenizer.pad(
            labels,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        return {
            "input_values": input_batch["input_values"],
            "labels": labels
        }

# Initialize collator
data_collator = CustomCTCCollator(processor=processor)

#Fine tune
# Update training arguments (adjust learning rate for Conformer)
training_args = TrainingArguments(
    output_dir="../models/conformer_fine_tuned_model",
    per_device_train_batch_size=4,  # Reduced for Conformer's larger size
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=1e-5,  # Lower learning rate for fine-tuning
    warmup_steps=1000,
    max_steps=5000,
    fp16=True,
    push_to_hub=False,
    remove_unused_columns=True  # Must be True with data collator
)

# Load WER metric
wer_metric = load("wer")

# Define metrics computation function
def compute_metrics(pred):
    # Get predictions and labels
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # Decode predictions
    pred_str = processor.batch_decode(pred_ids)
    
    # Decode labels (replace -100 with pad_token_id)
    label_ids = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

#Trainer with metrics computation
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

# Print formatted results
print(f"\nTest WER: {test_results['test_wer'] * 100:.2f}%")
print(f"Test Loss: {test_results['test_loss']:.4f}")
