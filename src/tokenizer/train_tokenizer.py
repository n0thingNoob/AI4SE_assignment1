"""
Train a Byte-level BPE tokenizer on Python code
"""
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


def train_tokenizer(corpus_path, output_dir, vocab_size=1000):
    """
    Train a byte-level BPE tokenizer
    
    Args:
        corpus_path: Path to the text corpus file
        output_dir: Directory to save the tokenizer
        vocab_size: Size of the vocabulary
    """
    
    print(f"Training tokenizer on: {corpus_path}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize a tokenizer with Byte-level BPE
    tokenizer = Tokenizer(models.BPE())
    
    # Set up pre-tokenization (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Set up the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True,
    )
    
    # Train the tokenizer
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    # Set up post-processing (add special tokens)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Save the tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    print(f"Tokenizer saved to: {tokenizer_path}")
    
    # Test the tokenizer
    test_code = "if x > 0:"
    encoding = tokenizer.encode(test_code)
    print(f"\nTest encoding: '{test_code}'")
    print(f"Tokens: {encoding.tokens}")
    print(f"IDs: {encoding.ids}")
    
    return tokenizer_path


if __name__ == "__main__":
    corpus_path = "/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/data/raw/pretrain_corpus.txt"
    output_dir = "/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/data/processed"
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        print("Please run generate_samples.py first")
        exit(1)
    
    tokenizer_path = train_tokenizer(corpus_path, output_dir, vocab_size=1000)
    print("\nTokenizer training complete!")
