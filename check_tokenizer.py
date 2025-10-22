#!/usr/bin/env python3
"""Check tokenizer status."""

from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("artifacts/tokenizer")
    
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.additional_special_tokens}")
    print(f"<IFMASK> ID: {tokenizer.convert_tokens_to_ids('<IFMASK>')}")
    print(f"<ANS> ID: {tokenizer.convert_tokens_to_ids('<ANS>')}")
    print(f"UNK ID: {tokenizer.unk_token_id}")
    
    # Check if special tokens are properly configured
    mask_id = tokenizer.convert_tokens_to_ids('<IFMASK>')
    ans_id = tokenizer.convert_tokens_to_ids('<ANS>')
    
    if mask_id == tokenizer.unk_token_id:
        print("❌ <IFMASK> is UNK token - tokenizer needs retraining")
    else:
        print("✅ <IFMASK> is properly configured")
        
    if ans_id == tokenizer.unk_token_id:
        print("❌ <ANS> is UNK token - tokenizer needs retraining")
    else:
        print("✅ <ANS> is properly configured")

if __name__ == "__main__":
    main()



