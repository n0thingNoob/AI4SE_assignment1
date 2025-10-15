"""
Pre-train a simple CLM (Causal Language Model) on Python code
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm


class CodeDataset(Dataset):
    """Simple dataset for code snippets"""
    
    def __init__(self, tokenizer, corpus_path, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read corpus
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        # Tokenize the entire corpus
        self.encodings = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding="max_length"
        )
        
        # Split into chunks
        self.input_ids = self.encodings['input_ids']
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.input_ids[idx]
        }


def train_clm(tokenizer_path, corpus_path, output_dir, num_epochs=3, batch_size=2):
    """
    Train a simple CLM model
    
    Args:
        tokenizer_path: Path to the tokenizer JSON file
        corpus_path: Path to the training corpus
        output_dir: Directory to save the model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
    """
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>"
    )
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Create model configuration
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print("Creating model...")
    model = GPT2LMHeadModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dataset
    print(f"Loading corpus from: {corpus_path}")
    dataset = CodeDataset(tokenizer, corpus_path, max_length=128)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-4)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track loss
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "clm_model")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Test generation
    print("\nTesting model generation...")
    model.eval()
    test_prompt = "def check_positive(x):\n    if"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")
    
    return model_path


if __name__ == "__main__":
    tokenizer_path = "/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/data/processed/tokenizer.json"
    corpus_path = "/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/data/raw/pretrain_corpus.txt"
    output_dir = "/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/models"
    
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please run train_tokenizer.py first")
        exit(1)
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus not found at {corpus_path}")
        print("Please run generate_samples.py first")
        exit(1)
    
    model_path = train_clm(tokenizer_path, corpus_path, output_dir, num_epochs=3, batch_size=2)
    print("\nPre-training complete!")
