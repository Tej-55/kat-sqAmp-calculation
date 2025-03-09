# Import necessary libraries
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from Dataset import AmplitudeDataset
from utils import evaluate_sequence_accuracy, plot_training_history, train_model
import argparse


def input_args():
    parser = argparse.ArgumentParser(description='Train a T5 model for amplitude data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    #parser.add_argument('--max_length', type=int, default=512, help='Maximum input length for the model')
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save the trained model')
    parser.add_argument('--source_dir', type=str, default='data', help='Directory containing the data files')
    return parser.parse_args()

def main():
    
    # Parse input arguments
    args = input_args()
    
    # Load the data back from the pickle file
    source_dir = args.source_dir
    train_df = pd.read_pickle(os.path.join(source_dir, "train.pkl"))
    val_df = pd.read_pickle(os.path.join(source_dir, "val.pkl"))
    test_df = pd.read_pickle(os.path.join(source_dir, "test.pkl"))
    
    
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Add special tokens for mathematical expressions
    special_tokens = ['*', '/', '+', '-', '^', '(', ')', '{', '}', '_', 'gamma', 'sigma', 'e^2']
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Vocabulary size after adding special tokens: {len(tokenizer)}")
    print(f"Special tokens added: {special_tokens}")
    
    # Create datasets
    train_dataset = AmplitudeDataset(
        train_df['tokenized_amplitude'].tolist(),
        train_df['tokenized_squared_amplitude'].tolist(),
        tokenizer
    )

    val_dataset = AmplitudeDataset(
        val_df['tokenized_amplitude'].tolist(),
        val_df['tokenized_squared_amplitude'].tolist(),
        tokenizer
    )

    test_dataset = AmplitudeDataset(
        test_df['tokenized_amplitude'].tolist(),
        test_df['tokenized_squared_amplitude'].tolist(),
        tokenizer
    )

    # Create data loaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Train the model
    # Set epochs to a small number for initial testing, increase for better results
    trained_model, history = train_model(model, train_loader, val_loader, epochs=args.num_epochs, lr=args.lr)

    # Plot training history
    plot_training_history(os.path.join(args.model_dir, 'training_history.png'))
    
    # Evaluate on test set
    test_accuracy, sample_predictions, sample_targets = evaluate_sequence_accuracy(trained_model, test_loader, tokenizer)
    print(f"Test sequence accuracy: {test_accuracy:.4f}")

    # Display some example predictions
    print("\nSample predictions vs targets:")
    for i, (pred, target) in enumerate(zip(sample_predictions, sample_targets)):
        print(f"\nExample {i+1}:")
        print(f"Prediction: {pred[:100]}..." if len(pred) > 100 else f"Prediction: {pred}")
        print(f"Target: {target[:100]}..." if len(target) > 100 else f"Target: {target}")
        print(f"Correct: {pred == target}")

    # Save the model and tokenizer
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(args.model_dir, 'amplitude_model.pth'))
    tokenizer.save_pretrained(os.path.join(args.model_dir, 'amplitude_tokenizer'))

    print("Model and tokenizer saved successfully!")
    
if __name__ == "__main__":
    main()