from torch.utils.data import Dataset
from typing import List, Callable

# Define the dataset class for amplitude data
class AmplitudeDataset(Dataset):
    def __init__(self, amplitudes: List, squared_amplitudes: List, tokenizer: Callable, max_length: int = 512):
        self.tokenizer = tokenizer
        self.amplitudes = amplitudes
        self.squared_amplitudes = squared_amplitudes
        self.max_length = max_length
        
    def __len__(self):
        return len(self.amplitudes)
    
    def __getitem__(self, idx: int):
        amplitude = ' '.join(self.amplitudes[idx])
        squared_amplitude = ' '.join(self.squared_amplitudes[idx])
        
        # Tokenize inputs
        input_encoding = self.tokenizer(
            amplitude,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        target_encoding = self.tokenizer(
            squared_amplitude,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding.input_ids.squeeze(),
            'attention_mask': input_encoding.attention_mask.squeeze(),
            'labels': target_encoding.input_ids.squeeze()
        }