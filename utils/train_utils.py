import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist
from utils.tokenizer import PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX



def train_model(model, train_loader, val_loader, epochs=3, lr=5e-5, args=None):

    device = torch.device(f'cuda:{args.local_rank}' if args.distributed else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.local_rank in [-1, 0]:
        print(f"Using device: {device}")

    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        # Set epoch for distributed sampler
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=args.local_rank not in [-1, 0])):            
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
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0 and args.local_rank in [-1, 0]:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Average loss across all processes
        if args.distributed:
            train_loss_tensor = torch.tensor(train_loss).to(device)
            dist.all_reduce(train_loss_tensor)
            train_loss = train_loss_tensor.item() / args.world_size
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=args.local_rank not in [-1, 0]):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        # Average validation loss across all processes
        if args.distributed:
            val_loss_tensor = torch.tensor(val_loss).to(device)
            dist.all_reduce(val_loss_tensor)
            val_loss = val_loss_tensor.item() / args.world_size
            
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        if args.local_rank in [-1, 0]:
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
    if save_path:
        plt.savefig(save_path)
    plt.show()


def evaluate_sequence_accuracy(model, data_loader, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Check if model is wrapped with DDP and access the base model if needed
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    all_predictions = []
    all_targets = []
    total_tokens = 0
    correct_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512
            )
            
            # For source sequences
            predictions = [tokenizer.tgt_decode(pred, skip_special_tokens=True) for pred in outputs]
            # For target sequences
            targets = [tokenizer.tgt_decode(label, skip_special_tokens=True) for label in labels]
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Calculate token accuracy (ignoring PAD and UNK tokens)
            for pred, label in zip(outputs, labels):
                # Convert to lists and trim to shorter length
                pred_list = pred.cpu().tolist()
                label_list = label.cpu().tolist()
                
                if pred_list and pred_list[0] in [PAD_IDX, BOS_IDX]:  
                        pred_list = pred_list[1:]
                
                min_len = min(len(pred_list), len(label_list))
                
                # Compare tokens
                for p, t in zip(pred_list[:min_len], label_list[:min_len]):
                    # Skip PAD and UNK tokens
                    if t not in [PAD_IDX, UNK_IDX, EOS_IDX, BOS_IDX]:
                        total_tokens += 1
                        if p == t:
                            correct_tokens += 1
            
            # Print progress
            if (batch_idx + 1) % 20 == 0:
                print(f"Evaluated {batch_idx+1}/{len(data_loader)} batches")
    
    # Calculate sequence accuracy
    exact_matches = sum(1 for pred, target in zip(all_predictions, all_targets) if pred == target)
    sequence_accuracy = exact_matches / len(all_targets)
    
    # Calculate token accuracy
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    
    # Return accuracy metrics and some examples for inspection
    return sequence_accuracy, token_accuracy, all_predictions[:25], all_targets[:25]


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train_titans_model(model, train_loader, val_loader, epochs=3, lr=5e-5, args=None):
    device = torch.device(f'cuda:{args.local_rank}' if args.distributed else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.local_rank in [-1, 0]:
        print(f"Using device: {device}")

    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(epochs):
        # Set epoch for distributed sampler
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=args.local_rank not in [-1, 0])):            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Prepare inputs for MAC Transformer
            # Combine input_ids and labels for teacher forcing
            combined_input = torch.cat([input_ids, labels[:, :-1]], dim=1)

            # Forward pass
            loss = model(
                combined_input,
                return_loss=True
            )

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Print progress every 10 batches
            if (batch_idx + 1) % 50 == 0 and args.local_rank in [-1, 0]:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Average loss across all processes
        if args.distributed:
            train_loss_tensor = torch.tensor(train_loss).to(device)
            dist.all_reduce(train_loss_tensor)
            train_loss = train_loss_tensor.item() / args.world_size

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=args.local_rank not in [-1, 0]):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Prepare inputs for MAC Transformer
                combined_input = torch.cat([input_ids, labels[:, :-1]], dim=1)

                # Forward pass
                loss = model(
                    combined_input,
                    return_loss=True
                )

                val_loss += loss.item()
        # Average validation loss across all processes
        if args.distributed:
            val_loss_tensor = torch.tensor(val_loss).to(device)
            dist.all_reduce(val_loss_tensor)
            val_loss = val_loss_tensor.item() / args.world_size

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        if args.local_rank in [-1, 0]:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train loss: {avg_train_loss:.4f}")
            print(f"Validation loss: {avg_val_loss:.4f}")
            print("-" * 50)

    return model, history

def evaluate_titans_sequence_accuracy(model, test_loader, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Generate predictions
            predictions = model.sample(
                input_ids,
                seq_len=input_ids.shape[1] + 300,  # Add extra length for generation
                temperature=0.3,
                use_cache=True
            )

            # Decode predictions and targets
            for i in range(len(input_ids)):
                pred_text = tokenizer.decode(predictions[i], skip_special_tokens=True)
                target_text = tokenizer.decode(labels[i], skip_special_tokens=True)

                all_predictions.append(pred_text)
                all_targets.append(target_text)

                # Calculate token accuracy
                pred_list = predictions[i].cpu().tolist()
                label_list = labels[i].cpu().tolist()
                
                # Remove start token if present
                if pred_list and pred_list[0] in [PAD_IDX, BOS_IDX]:
                    pred_list = pred_list[1:]
                
                # Get minimum length for comparison
                min_len = min(len(pred_list), len(label_list))
                
                # Compare tokens
                for p, t in zip(pred_list[:min_len], label_list[:min_len]):
                    # Skip special tokens
                    if t not in [PAD_IDX, UNK_IDX, EOS_IDX, BOS_IDX]:
                        total_tokens += 1
                        if p == t:
                            correct_tokens += 1

            # Print progress
            if (batch_idx + 1) % 20 == 0:
                print(f"Evaluated {batch_idx+1}/{len(test_loader)} batches")

    # Calculate sequence accuracy
    exact_matches = sum(1 for pred, target in zip(all_predictions, all_targets) if pred == target)
    sequence_accuracy = exact_matches / len(all_targets)
    
    # Calculate token accuracy
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0

    # Return accuracy and some examples for inspection
    return sequence_accuracy, token_accuracy, all_predictions[:25], all_targets[:25]