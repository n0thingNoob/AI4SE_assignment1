"""
Run minimum test case to verify the entire pipeline
"""
import sys
import os

# Add src to path
sys.path.insert(0, '/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/src')

from data.generate_samples import generate_sample_data
from tokenizer.train_tokenizer import train_tokenizer
from modeling.pretrain_clm import train_clm


def run_minimum_test():
    """Run a complete minimum test of the pipeline"""
    
    print("="*60)
    print("MINIMUM PIPELINE TEST")
    print("="*60)
    
    # Step 1: Generate sample data
    print("\n[STEP 1] Generating sample data...")
    try:
        data_dir = "/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/data/raw"
        corpus_path = generate_sample_data(data_dir)
        print("✓ Sample data generation successful")
    except Exception as e:
        print(f"✗ Sample data generation failed: {e}")
        return False
    
    # Step 2: Train tokenizer
    print("\n[STEP 2] Training tokenizer...")
    try:
        output_dir = "/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/data/processed"
        tokenizer_path = train_tokenizer(corpus_path, output_dir, vocab_size=1000)
        print("✓ Tokenizer training successful")
    except Exception as e:
        print(f"✗ Tokenizer training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Train CLM model
    print("\n[STEP 3] Training CLM model...")
    try:
        model_output_dir = "/home/runner/work/AI4SE_assignment1/AI4SE_assignment1/models"
        model_path = train_clm(
            tokenizer_path,
            corpus_path,
            model_output_dir,
            num_epochs=2,  # Reduced for quick test
            batch_size=2
        )
        print("✓ CLM training successful")
    except Exception as e:
        print(f"✗ CLM training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - Data: {data_dir}")
    print(f"  - Tokenizer: {tokenizer_path}")
    print(f"  - Model: {model_path}")
    print("\nThe pipeline is working correctly!")
    
    return True


if __name__ == "__main__":
    success = run_minimum_test()
    sys.exit(0 if success else 1)
