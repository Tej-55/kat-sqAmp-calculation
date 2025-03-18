# Import necessary libraries
import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from utils.data_utils import get_data, load_data
from utils.tokenizer import create_tokenizer
from utils.train_utils import evaluate_sequence_accuracy, plot_training_history, train_model
import argparse


def input_args():
    parser = argparse.ArgumentParser(description='Train a T5 model for amplitude data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum input length for the model')
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save the trained model')
    parser.add_argument('--source_dir', type=str, default='data', help='Directory containing the data files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
def main():    
    # Parse input arguments
    args = input_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    model_dir = args.model_dir
    source_dir = args.source_dir
    max_length = args.max_length
    seed = args.seed
    
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Load the data 
    file_paths = [f"{source_dir}/QED-2-to-2-diag-TreeLevel-{i}.txt" for i in range(0, 10)]
    df = load_data(file_paths)    
    
    # Initialize tokenizer and model
    tokenizer, src_vocab, tgt_vocab, src_itos, tgt_itos = create_tokenizer(df, index_pool_size=100, momentum_pool_size=100)
    data = get_data(df, tokenizer, src_vocab, tgt_vocab, max_len=max_length)
    
    train_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data['valid'], batch_size=batch_size)
    test_loader = DataLoader(data['test'], batch_size=batch_size)
    
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.resize_token_embeddings(len(src_vocab))

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Train the model
    # Set epochs to a small number for initial testing, increase for better results
    trained_model, history = train_model(model, train_loader, val_loader, epochs=num_epochs, lr=lr)

    # Plot training history
    os.makedirs(args.model_dir, exist_ok=True)
    plot_training_history(history, os.path.join(model_dir, 'training_history.png'))
    
    # Evaluate on test set
    test_seq_accuracy, test_token_accuracy, sample_predictions, sample_targets = evaluate_sequence_accuracy(trained_model, test_loader, tokenizer)
    print(f"Test sequence accuracy: {test_seq_accuracy:.4f}")
    print(f"Test token accuracy: {test_token_accuracy:.4f}")

    # Display some example predictions
    print("\nSample predictions vs targets:")
    for i, (pred, target) in enumerate(zip(sample_predictions, sample_targets)):
        print(f"\nExample {i+1}:")
        print(f"Prediction: {pred[:100]}..." if len(pred) > 100 else f"Prediction: {pred}")
        print(f"Target: {target[:100]}..." if len(target) > 100 else f"Target: {target}")
        print(f"Correct: {pred == target}")

    # Save the model and tokenizer
    torch.save(trained_model.state_dict(), os.path.join(args.model_dir, 'amplitude_model.pth'))
    #tokenizer.save_pretrained(os.path.join(args.model_dir, 'amplitude_tokenizer'))

    print("Model and tokenizer saved successfully!")
    
if __name__ == "__main__":
    main()