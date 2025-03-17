import torch
from torch.utils.data import Dataset
from typing import List, Callable
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.tokenizer import BOS_IDX, EOS_IDX, PAD_IDX

def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' : ')
                if len(parts) == 4:
                    event_type, diagram, amplitude, squared_amplitude = parts
                    data.append({
                        'event_type': event_type,
                        'diagram': diagram,
                        'amplitude': amplitude.strip(),
                        'squared_amplitude': squared_amplitude.strip()
                    })
    return pd.DataFrame(data)

def get_data(data, tokenizer, src_vocab, tgt_vocab, max_len, split = [0.8, 0.1, 0.1]):
    df_train, temp_df = train_test_split(data, test_size=1-split[0], random_state=42)
    df_valid, df_test = train_test_split(temp_df, test_size=(split[1]/(split[1]+split[2])), random_state=42)
    train = AmplitudeDataset(df_train, tokenizer, src_vocab, tgt_vocab, max_len=max_len)
    test = AmplitudeDataset(df_test, tokenizer, src_vocab, tgt_vocab, max_len=max_len)
    valid = AmplitudeDataset(df_valid, tokenizer, src_vocab, tgt_vocab, max_len=max_len)

    return {'train': train, 'test': test, 'valid': valid}

# Define the dataset class for amplitude data
class AmplitudeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: Callable, max_length: int = 512, src_vocab=None, tgt_vocab=None):
        super(AmplitudeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.amplitudes = data['amplitude'].tolist()
        self.squared_amplitudes = data['squared_amplitude'].tolist()
        self.max_length = max_length
        self.tgt_tokenize = tokenizer.tgt_tokenize
        self.src_tokenize = tokenizer.src_tokenize
        self.bos_token = torch.tensor([BOS_IDX], dtype=torch.int64)
        self.eos_token = torch.tensor([EOS_IDX], dtype=torch.int64)
        self.pad_token = torch.tensor([PAD_IDX], dtype=torch.int64)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def __len__(self):
        return len(self.amplitudes)
    
    def __getitem__(self, idx: int):
        src_tokenized = self.src_tokenize(self.amplitudes[idx])
        tgt_tokenized = self.tgt_tokenize(self.squared_amplitudes[idx])
        src_ids = self.src_vocab(src_tokenized)
        tgt_ids = self.tgt_vocab(tgt_tokenized)

        # Calculate padding tokens needed
        src_max_len = self.max_length
        tgt_max_len = self.max_length
        
        enc_num_padding_tokens = src_max_len - len(src_ids) - 2  # -2 for BOS and EOS
        dec_num_padding_tokens = tgt_max_len - len(tgt_ids) - 1  # -1 for BOS

        # Handle truncation if needed
        if enc_num_padding_tokens < 0:
            src_ids = src_ids[:src_max_len-2]
            enc_num_padding_tokens = 0
        if dec_num_padding_tokens < 0:
            tgt_ids = tgt_ids[:tgt_max_len-1]
            dec_num_padding_tokens = 0

        # Create tensors with appropriate padding
        src_tensor = torch.cat(
            [
                self.bos_token,
                torch.tensor(src_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([PAD_IDX] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        tgt_tensor = torch.cat(
            [
                self.bos_token,
                torch.tensor(tgt_ids, dtype=torch.int64),
                torch.tensor([PAD_IDX] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(tgt_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([PAD_IDX] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create attention masks
        attention_mask = (src_tensor != PAD_IDX).int()
        
        # For T5, we need to return a dictionary with the expected keys
        return {
            'input_ids': src_tensor,
            'attention_mask': attention_mask,
            'labels': label
        }