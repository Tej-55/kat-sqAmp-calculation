import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train_utils import generate_square_subsequent_mask

class TitansEncoder(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        nhead: int = 8, 
        num_layers: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: int = 0.1
        ):
        super(TitansEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(1024, d_model)  # Position encoding
        
        # Transformer encoder layers with additional components
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # TITANS-specific components: Symbolic Reasoning Module
        self.symbolic_reasoning = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.d_model = d_model
        
    def forward(self, src_mask=None, src_key_padding_mask=None):
        # Create position indices
        positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
        
        # Embed tokens and positions
        src = self.embedding(src) * (self.d_model ** 0.5)
        pos_encoding = self.pos_encoder(positions)
        src = src + pos_encoding
        
        # Transformer encoding
        memory = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Apply symbolic reasoning module
        enhanced_memory = memory + self.symbolic_reasoning(memory)
        
        return enhanced_memory

class TitansDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TitansDecoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(1024, d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # TITANS-specific components: Pattern Recognition Module
        self.pattern_recognition = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Create position indices
        positions = torch.arange(0, tgt.size(1), device=tgt.device).unsqueeze(0).expand(tgt.size(0), -1)
        
        # Embed tokens and positions
        tgt = self.embedding(tgt) * (self.d_model ** 0.5)
        pos_encoding = self.pos_encoder(positions)
        tgt = tgt + pos_encoding
        
        # Transformer decoding
        output = self.transformer_decoder(
            tgt, memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Apply pattern recognition module
        enhanced_output = output + self.pattern_recognition(output)
        
        # Project to vocabulary
        return self.output_projection(enhanced_output)

class TitansModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, start_token_id=None):
        super(TitansModel, self).__init__()
        
        self.encoder = TitansEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.decoder = TitansDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.start_token_id = start_token_id
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        
        output = self.decoder(
            tgt, memory, 
            tgt_mask=tgt_mask, 
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return output
    
    def generate(self, src, max_length, bos_token_id, eos_token_id, 
                 src_mask=None, src_key_padding_mask=None):
        batch_size = src.size(0)
        device = src.device
        
        # Encode the source sequence
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        
        # Initialize the target sequence with BOS token
        tgt = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=device)
        
        for i in range(max_length - 1):
            # Create masks for the target sequence
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Decode the next token
            output = self.decoder(
                tgt, memory, 
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            # Get the next token prediction
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the next token to the target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check if all sequences have reached EOS
            if (tgt == eos_token_id).any(dim=1).all():
                break
                
        return tgt




