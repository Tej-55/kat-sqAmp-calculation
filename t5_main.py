# Import necessary libraries
import os
import pandas as pd
import numpy as np
import random
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.data_utils import get_data, load_data
from utils.tokenizer import create_tokenizer
from utils.train_utils import evaluate_sequence_accuracy, plot_training_history, train_model

def input_args():
    parser = argparse.ArgumentParser(description='Train a T5 model for amplitude data')
    # Existing arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum input length for the model')
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save the trained model')
    parser.add_argument('--source_dir', type=str, default='data', help='Directory containing the data files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # New distributed training arguments
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for distributed training')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')
    
    return parser.parse_args()

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank
    )
    dist.barrier()


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
    
    # Initialize distributed training
    if args.local_rank != -1:
        init_distributed_mode(args)
    else:
        args.distributed = False
    
    # Set random seed for reproducibility
    set_seed(args.seed + args.local_rank if args.distributed else args.seed)
    
    # Load the data 
    file_paths = [f"{source_dir}/QED-2-to-2-diag-TreeLevel-{i}.txt" for i in range(0, 10)]
    df = load_data(file_paths)    
    
    # Initialize tokenizer and model
    tokenizer, src_vocab, tgt_vocab, src_itos, tgt_itos = create_tokenizer(df, index_pool_size=100, momentum_pool_size=100)
    data = get_data(df, tokenizer, src_vocab, tgt_vocab, max_len=max_length)
    
    # Create distributed samplers for training and validation
    if args.distributed:
        train_sampler = DistributedSampler(data['train'])
        val_sampler = DistributedSampler(data['valid'], shuffle=False)
        test_sampler = DistributedSampler(data['test'], shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    # Create data loaders with appropriate samplers
    train_loader = DataLoader(
        data['train'], 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        data['valid'], 
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        data['test'], 
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True
    )
    
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.resize_token_embeddings(len(src_vocab))
    
    # Move model to appropriate device
    device = torch.device(f'cuda:{args.local_rank}' if args.distributed else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Wrap model with DDP if using distributed training
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    if args.local_rank in [-1, 0]:
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
    
    # Train the model
    # Set epochs to a small number for initial testing, increase for better results
    trained_model, history = train_model(model, train_loader, val_loader, epochs=num_epochs, lr=lr, args=args)
    
    # Only the main process evaluates and saves the model
    if args.local_rank in [-1, 0]:
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
            print(f"Prediction: {pred[:250]}..." if len(pred) > 250 else f"Prediction: {pred}")
            print(f"Target: {target[:250]}..." if len(target) > 250 else f"Target: {target}")
            print(f"Correct: {pred == target}")

        # Save the model and tokenizer
        torch.save(trained_model.state_dict(), os.path.join(args.model_dir, 'amplitude_model.pth'))
        print("Model saved successfully!")
    
if __name__ == "__main__":
    main()