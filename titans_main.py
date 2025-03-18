# Import necessary libraries
import os
import pandas as pd
import numpy as np
import random
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.data_utils import AmplitudeDataset, get_data, load_data
from titans.memory_models import MemoryMLP, ResidualNorm
from titans.neural_memory import NeuralMemory
from titans.mac_transformer import MemoryAsContextTransformer
from utils.data_utils import AmplitudeDataset
from utils.tokenizer import create_tokenizer
from utils.train_utils import evaluate_titans_sequence_accuracy, plot_training_history, train_titans_model


def input_args():
    parser = argparse.ArgumentParser(description='Train a T5 model for amplitude data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum input length for the model')
    parser.add_argument('--model_dir', type=str, default='model_titans', help='Directory to save the trained model')
    parser.add_argument('--source_dir', type=str, default='data', help='Directory containing the data files')
    parser.add_argument('--segment_len', type=int, default=64, help='Segment length for the TITANS model')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--depth', type=int, default=2, help='Depth of the model')
    parser.add_argument('--heads', type=int, default=4, help='Number of heads in the model')
    parser.add_argument('--depth_mem', type=int, default=3, help='Depth of the Memory Module in the model')
    parser.add_argument('--expansion_factor', type=float, default=2.0, help='Expansion factor for the Memory Module')
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
    max_length = args.max_length
    source_dir = args.source_dir
    model_dir = args.model_dir
    segment_len = args.segment_len
    dim = args.dim
    depth = args.depth
    heads = args.heads
    depth_mem = args.depth_mem
    expansion_factor = args.expansion_factor
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
    
    # Train the model
    # Initialize TITANS model
    neural_memory_model = MemoryMLP(
        dim=dim,
        depth=depth_mem,
        expansion_factor=expansion_factor
    )
    titans_model = MemoryAsContextTransformer(
        num_tokens=len(tokenizer),
        dim=dim,
        depth=depth,
        segment_len=segment_len,
        neural_memory_segment_len=segment_len,
        neural_memory_add_value_residual=True,
        neural_mem_gate_attn_output=True,
        heads=heads,
        neural_memory_model=neural_memory_model,
        neural_memory_qkv_receives_diff_views=True,
        sliding_window_attn=True
    )
    
    # Move model to appropriate device
    device = torch.device(f'cuda:{args.local_rank}' if args.distributed else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    titans_model.to(device)
    
    # Wrap model with DDP if using distributed training
    if args.distributed:
        titans_model = DDP(titans_model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    if args.local_rank in [-1, 0]:
        print(f"Model parameters: {sum(p.numel() for p in titans_model.parameters())}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
    
    # Set epochs to a small number for initial testing, increase for better results
    trained_titans_model, history = train_titans_model(
        titans_model, 
        tokenizer,
        train_loader, 
        val_loader, 
        epochs=num_epochs, 
        lr=lr,
        args=args
    )
    
    # Only the main process evaluates and saves the model
    if args.local_rank in [-1, 0]:
        # Plot training history
        os.makedirs(args.model_dir, exist_ok=True)
        plot_training_history(history, os.path.join(args.model_dir, 'training_history.png'))
        
        # Evaluate on test set
        test_seq_accuracy, test_token_accuracy, sample_predictions, sample_targets = evaluate_titans_sequence_accuracy(
            trained_titans_model, 
            tokenizer, 
            test_loader
        )
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
        torch.save(trained_titans_model.state_dict(), os.path.join(args.model_dir, 'titans_model.pt'))
        # tokenizer.save_pretrained(os.path.join(args.model_dir, 'titans_tokenizer'))

    print("Model saved successfully!")
    
if __name__ == "__main__":
    main()