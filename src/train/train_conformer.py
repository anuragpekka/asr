# With Predict function

import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import os
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from jiwer import wer
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pdb

# Configuration
@dataclass
class ConformerConfig:
    vocab_size: int = 1000  # Size of vocabulary (will be set based on actual data)
    encoder_dim: int = 128 #256
    # num_encoder_layers: int = 12
    # num_attention_heads: int = 4
    # encoder_dim=128,  # Reduced from 256
    num_encoder_layers=4  # Reduced from 12
    num_attention_heads=4
    feed_forward_expansion_factor: int = 4
    conv_expansion_factor: int = 2
    input_dropout_p: float = 0.1
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    conv_kernel_size: int = 31
    half_step_residual: bool = True
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160



class AudioProcessor:
    def __init__(self, config: ConformerConfig):
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length
        )
        
    def process_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.config.sample_rate)
        # Ensure mono audio
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        mel_spec = self.mel_transform(waveform)  # Shape: [1, n_mels, time]
        log_mel = torch.log1p(mel_spec)
        return log_mel.squeeze(0)  # Shape: [n_mels, time]

class TextProcessor:
    def __init__(self):
        self.char2idx = {'<blank>': 0}  # Reserve index 0 for blank
        self.idx2char = {0: '<blank>'}
        self.vocab_size = 1
        
    def fit(self, texts: List[str]):
        unique_chars = set()
        for text in texts:
            unique_chars.update(list(text))
        
        # Assign indices starting from 1
        for idx, char in enumerate(sorted(unique_chars), start=1):
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        self.vocab_size = len(self.char2idx)
        
    def encode(self, text: str) -> List[int]:
        return [self.char2idx[c] for c in text]
    
    def decode(self, indices: List[int]) -> str:
        return ''.join([self.idx2char[idx] for idx in indices if idx != 0])  # Skip blank tokens

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x['audio_features'].shape[1], reverse=True)
    
    audio_features = [item['audio_features'] for item in batch]
    text_indices_list = [item['text_indices'] for item in batch]
    texts = [item['text'] for item in batch]
    
    # Process audio features
    audio_lengths = torch.LongTensor([feature.shape[1] for feature in audio_features])
    max_audio_len = max(feature.shape[1] for feature in audio_features)
    padded_audio = []
    for feature in audio_features:
        pad_size = max_audio_len - feature.shape[1]
        padded_feature = F.pad(feature, (0, pad_size), "constant", 0)
        padded_audio.append(padded_feature)
    audio_features = torch.stack(padded_audio)
    
    # Process text indices for CTC (flatten and track lengths)
    text_lengths = torch.LongTensor([len(t) for t in text_indices_list])
    text_indices = torch.cat(text_indices_list)  # Flatten into 1D
    
    return {
        'audio_features': audio_features,
        'audio_lengths': audio_lengths,
        'text_indices': text_indices,
        'text_lengths': text_lengths,
        'text': texts
    }

class HindiASRDataset(Dataset):
    def __init__(self, transcript_file: str, audio_dir: str, config: ConformerConfig, text_processor: TextProcessor):
        self.data = []
        self.audio_dir = audio_dir
        self.audio_processor = AudioProcessor(config)
        self.text_processor = text_processor
        
        # Validate audio directory exists
        if not os.path.exists(audio_dir):
            raise ValueError(f"Audio directory not found: {audio_dir}")
            
        # Validate transcript file exists
        if not os.path.exists(transcript_file):
            raise ValueError(f"Transcript file not found: {transcript_file}")
        
        # Load and validate transcript file
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Validate line format
                    if '|' not in line:
                        raise ValueError(f"Invalid format in line {line_num}: {line.strip()}")
                    
                    audio_file, text = line.strip().split('|')
                    audio_path = os.path.join(audio_dir, audio_file)
                    
                    # Validate audio file exists
                    if not os.path.exists(audio_path):
                        print(f"Warning: Audio file not found: {audio_path}")
                        continue
                    
                    self.data.append({
                        'audio_path': audio_path,
                        'text': text
                    })
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            audio_features = self.audio_processor.process_audio(item['audio_path'])
            text_indices = self.text_processor.encode(item['text'])
            return {
                'audio_features': audio_features,
                'text_indices': torch.tensor(text_indices),
                'text': item['text']
            }
        except Exception as e:
            print(f"Error processing item {idx}, file {item['audio_path']}: {str(e)}")
            # Return a dummy item of the correct format
            return {
                'audio_features': torch.zeros(self.audio_processor.config.n_mels, 100),
                'text_indices': torch.tensor([0]),
                'text': ''
            }

# Conformer Modules
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.encoder_dim,
            config.num_attention_heads,
            dropout=config.attention_dropout_p,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.encoder_dim)
        self.dropout = nn.Dropout(config.attention_dropout_p)
    
    def forward(self, x, mask=None):
        x = self.layer_norm(x)
        output, _ = self.attention(x, x, x, key_padding_mask=mask)
        return self.dropout(output)

class ConvModule(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.encoder_dim)
        self.conv1 = nn.Conv1d(
            config.encoder_dim,
            config.encoder_dim * config.conv_expansion_factor,
            1
        )
        self.glu = nn.GLU(dim=1)
        
        self.depth_conv = nn.Conv1d(
            config.encoder_dim,
            config.encoder_dim,
            config.conv_kernel_size,
            padding=(config.conv_kernel_size - 1) // 2,
            groups=config.encoder_dim
        )
        
        self.batch_norm = nn.BatchNorm1d(config.encoder_dim)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv1d(config.encoder_dim, config.encoder_dim, 1)
        self.dropout = nn.Dropout(config.conv_dropout_p)
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.glu(x)
        x = self.depth_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)

class FeedForwardModule(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.encoder_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.encoder_dim, config.encoder_dim * config.feed_forward_expansion_factor),
            nn.SiLU(),
            nn.Dropout(config.feed_forward_dropout_p),
            nn.Linear(config.encoder_dim * config.feed_forward_expansion_factor, config.encoder_dim),
            nn.Dropout(config.feed_forward_dropout_p)
        )
    
    def forward(self, x):
        x = self.layer_norm(x)
        return self.feed_forward(x)

class ConformerBlock(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.ff1 = FeedForwardModule(config)
        self.self_attention = MultiHeadedSelfAttention(config)
        self.conv = ConvModule(config)
        self.ff2 = FeedForwardModule(config)
        self.layer_norm = nn.LayerNorm(config.encoder_dim)
        
    def forward(self, x, mask=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.self_attention(x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.layer_norm(x)

class ConformerASR(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        
        # Modified input projection to handle the correct dimensions
        self.input_projection = nn.Linear(config.n_mels, config.encoder_dim)
        self.input_dropout = nn.Dropout(p=config.input_dropout_p)
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(config) for _ in range(config.num_encoder_layers)
        ])
        
        self.output_projection = nn.Linear(config.encoder_dim, config.vocab_size)
    
    def forward(self, inputs, input_lengths=None):
        # inputs shape: [batch, n_mels, time]
        # Transpose to [batch, time, n_mels] for linear layer
        x = inputs.transpose(1, 2)
        
        # Project to encoder dimension
        x = self.input_projection(x)  # Shape: [batch, time, encoder_dim]
        x = self.input_dropout(x)
        
        # Create padding mask if lengths are provided
        mask = None
        if input_lengths is not None:
            max_len = x.size(1)
            batch_size = x.size(0)
            mask = torch.arange(max_len, device=x.device)[None, :] >= input_lengths[:, None]
        
        # Apply Conformer blocks
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x, mask)
        
        # Output projection
        x = self.output_projection(x)  # Shape: [batch, time, vocab_size]
        
        return x

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        inputs = batch['audio_features'].to(device)
        input_lengths = batch['audio_lengths'].to(device)
        targets = batch['text_indices'].to(device)
        target_lengths = batch['text_lengths'].to(device)
        
        outputs = model(inputs, input_lengths)  # Shape: [B, T, V]
        
        # Apply log softmax and permute for CTC
        log_probs = F.log_softmax(outputs, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)  # [T, B, V]
        
        # Compute CTC loss
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
       
        if idx % 10 == 0:  # Print progress every 10 batches
            print(f'Batch {idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, text_processor, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['audio_features'].to(device)
            input_lengths = batch['audio_lengths'].to(device)
            
            outputs = model(inputs, input_lengths)
            predictions = torch.argmax(outputs, dim=-1)
            
            for pred, target in zip(predictions, batch['text']):
                pred_text = text_processor.decode(pred.cpu().tolist())
                all_predictions.append(pred_text)
                all_targets.append(target)
    
    return wer(all_targets, all_predictions)

def train_model(config: ConformerConfig, train_file: str, val_file: str, audio_dir: str, 
                num_epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    best_model_path = r"C:\Users\DAI.STUDENTSDC\PycharmProjects\pythonProject1\deep learning\natural language processing\Practice\conformer\best_model.pth"
    
    # Initialize text processor
    text_processor = TextProcessor()
    with open(train_file, 'r', encoding='utf-8') as f:
        texts = [line.strip().split('|')[1] for line in f]
    text_processor.fit(texts)
    config.vocab_size = text_processor.vocab_size
    print(f"Vocabulary size: {config.vocab_size}")
    
    # Create datasets
    train_dataset = HindiASRDataset(train_file, audio_dir, config, text_processor)
    val_dataset = HindiASRDataset(val_file, audio_dir, config, text_processor)
    
    # Use custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Initialize model and training components
    model = ConformerASR(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Use CTCLoss instead of CrossEntropyLoss
    criterion = nn.CTCLoss(blank=0)  # Blank token at index 0
        
    best_wer = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_wer = evaluate(model, val_loader, text_processor, device)
        
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation WER: {val_wer:.4f}')
        
        if val_wer < best_wer:
            best_wer = val_wer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_wer': val_wer,
            }, best_model_path)
            
        print(f'Best WER: {best_wer:.4f}')

def ctc_decode(indices: List[int]) -> List[int]:
    """Collapse repeated tokens and remove blanks."""
    decoded = []
    previous = None
    for idx in indices:
        if idx == 0:  # blank token
            previous = None
            continue
        if idx != previous:
            decoded.append(idx)
            previous = idx
    return decoded

def predict(audio_path: str, model_path: str = 'best_model.pth',
            train_transcript_path: str = r'transcript2.txt'):

    # Initialize components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ConformerConfig()
    
    # Load text processor
    text_processor = TextProcessor()
    with open(train_transcript_path, 'r', encoding='utf-8') as f:
        texts = [line.strip().split('|')[1] for line in f]
    text_processor.fit(texts)
    config.vocab_size = text_processor.vocab_size
    
    # pdb.set_trace()
    # Load model
    model = ConformerASR(config).to(device)
    # checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Process audio
    audio_processor = AudioProcessor(config)
    features = audio_processor.process_audio(audio_path).unsqueeze(0).to(device)  # [1, n_mels, time]
    
    # Run inference
    with torch.no_grad():
        outputs = model(features)  # [1, time, vocab_size]
    
    # Decode output
    pred_indices = torch.argmax(outputs, dim=-1).squeeze().cpu().tolist()
    decoded_indices = ctc_decode(pred_indices)
    return text_processor.decode(decoded_indices)

if __name__ == "__main__":
    config = ConformerConfig()
        
    train_model(
        config=config,
        train_file=r'C:\Users\DAI.STUDENTSDC\PycharmProjects\pythonProject1\deep learning\natural language processing\Practice\conformer\transcript2.txt',
        val_file=r'C:\Users\DAI.STUDENTSDC\PycharmProjects\pythonProject1\deep learning\natural language processing\Practice\conformer\transcription-valid.txt',
        audio_dir=r'F:\ProjectData\DS\mucs\Hindi\audio-16k',
        # train_file=r'C:\Users\DAI.STUDENTSDC\PycharmProjects\pythonProject1\deep learning\natural language processing\Practice\conformer\text_ranscription1.txt',
        # val_file=r'C:\Users\DAI.STUDENTSDC\PycharmProjects\pythonProject1\deep learning\natural language processing\Practice\conformer\text_ranscription1-valid.txt',
        # audio_dir=r"F:\ProjectData\DS\GV_Dev_5h\GV_Dev_5h_full\Audio",
        num_epochs= 50,
        batch_size=32,
        learning_rate=0.001
    )

    prediction = predict(
        audio_path=r'C:\Users\DAI.STUDENTSDC\PycharmProjects\pythonProject1\deep learning\natural language processing\Practice\conformer\test\2880_086.wav',
        model_path=r'C:\Users\DAI.STUDENTSDC\PycharmProjects\pythonProject1\deep learning\natural language processing\Practice\conformer\best_model.pth',
        train_transcript_path=r'C:\Users\DAI.STUDENTSDC\PycharmProjects\pythonProject1\deep learning\natural language processing\Practice\conformer\transcript2.txt')
    
    print(f"Predicted Transcript: '{prediction}'")
    with open("prediction-confomer-asr.txt", "w", encoding="utf-8") as f:
        f.writelines(prediction)
