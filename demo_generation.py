"""
Demo script to use the trained model for code generation
"""
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


def generate_code(model_path, prompt, max_length=50, num_samples=3):
    """
    Generate code completions using the trained model
    
    Args:
        model_path: Path to the trained model directory
        prompt: Input code prompt
        max_length: Maximum length of generated sequence
        num_samples: Number of samples to generate
    """
    
    print(f"Loading model from: {model_path}")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded (device: {device})")
    print(f"\nPrompt: {prompt}")
    print("=" * 60)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate multiple samples
    with torch.no_grad():
        for i in range(num_samples):
            outputs = model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=2
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nSample {i+1}:")
            print(generated_text)
    
    print("=" * 60)


if __name__ == "__main__":
    model_path = "/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/models/clm_model"
    
    # Test with different prompts
    prompts = [
        "def check_value(x):\n    if",
        "def is_valid(data):\n    if",
        "def compare(a, b):\n    if",
    ]
    
    for prompt in prompts:
        print("\n" + "=" * 60)
        generate_code(model_path, prompt, max_length=40, num_samples=2)
        print()
