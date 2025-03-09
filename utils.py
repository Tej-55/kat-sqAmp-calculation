import torch
import torch.nn as nn
import matplotlib.pyplot as plt



def train_model(model, train_loader, val_loader, epochs=3, lr=5e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")
        print("-" * 50)
    
    return model, history

def plot_training_history(history, save_path=None):    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    if save_path:
        plt.savefig(save_path)
    
def evaluate_sequence_accuracy(model, data_loader, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512
            )
            
            # Decode predictions and targets
            predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs]
            targets = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Print progress
            if (batch_idx + 1) % 5 == 0:
                print(f"Evaluated {batch_idx+1}/{len(data_loader)} batches")
    
    # Calculate sequence accuracy
    exact_matches = sum(1 for pred, target in zip(all_predictions, all_targets) if pred == target)
    sequence_accuracy = exact_matches / len(all_targets)
    
    # Return accuracy and some examples for inspection
    return sequence_accuracy, all_predictions[:5], all_targets[:5]

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Training and evaluation functions for TITANS model
def train_titans_model(model, tokenizer, train_loader, val_loader, epochs=3, lr=5e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Create masks
            src_key_padding_mask = (attention_mask == 0)
            tgt_input = labels[:, :-1]  # Remove last token for input
            tgt_output = labels[:, 1:]  # Remove first token (BOS) for target
            
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                src=input_ids,
                tgt=tgt_input,
                src_mask=None,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=None
            )
            
            # Reshape outputs for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs, tgt_output)
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Create masks
                src_key_padding_mask = (attention_mask == 0)
                tgt_input = labels[:, :-1]
                tgt_output = labels[:, 1:]
                
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                
                # Forward pass
                outputs = model(
                    src=input_ids,
                    tgt=tgt_input,
                    src_mask=None,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=None
                )
                
                # Reshape outputs for loss calculation
                outputs = outputs.view(-1, outputs.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                # Calculate loss
                loss = criterion(outputs, tgt_output)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")

    return model, history

def evaluate_titans_sequence_accuracy(model, tokenizer, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate predictions
            predictions = model.generate(
                src=input_ids,
                max_length=512,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                src_key_padding_mask=(attention_mask == 0)
            )
            
            # Decode predictions and targets
            pred_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
            target_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            
            all_predictions.extend(pred_texts)
            all_targets.extend(target_texts)

    # Calculate sequence accuracy
    exact_matches = sum(1 for pred, target in zip(all_predictions, all_targets) if pred == target)
    sequence_accuracy = exact_matches / len(all_targets)

    return sequence_accuracy, all_predictions, all_targets