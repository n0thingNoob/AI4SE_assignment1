#!/usr/bin/env python3
"""
Custom data collators for fine-tuning.

Provides a collator that masks the prompt portion in labels,
so the model only trains on generating the target sequence.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForIfConditionPrediction:
    """
    Data collator for if-condition prediction task.
    
    Masks the prompt portion of input so loss is only computed on target tokens.
    Input format: "<prompt><answer>\n<target></s>"
    Labels: [-100, -100, ..., -100, target_token_1, target_token_2, ..., eos]
    """
    
    tokenizer: PreTrainedTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    answer_token_id: Optional[int] = None
    
    def __post_init__(self):
        # Get answer token ID
        if self.answer_token_id is None:
            self.answer_token_id = self.tokenizer.convert_tokens_to_ids('<answer>')
            if self.answer_token_id == self.tokenizer.unk_token_id:
                # For GPT-2, use EOS token as answer token
                print("Warning: <answer> token not found in tokenizer, using EOS token")
                self.answer_token_id = self.tokenizer.eos_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples.
        
        Each feature should have 'input_ids' and 'attention_mask'.
        The input_ids should contain the full sequence: prompt + <answer> + target.
        
        Args:
            features: List of dictionaries with 'input_ids' and 'attention_mask'
            
        Returns:
            Batch dictionary with input_ids, attention_mask, and labels
        """
        # Pad sequences
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        
        # Create labels by masking prompt portion
        input_ids = batch['input_ids'].clone()
        labels = input_ids.clone()
        
        # For each sequence, find the <answer> token and mask everything before it
        for i in range(len(labels)):
            # Find <answer> token position
            answer_positions = (input_ids[i] == self.answer_token_id).nonzero(as_tuple=True)[0]
            
            if len(answer_positions) > 0:
                # Mask everything up to and including the <answer> token
                answer_pos = answer_positions[0].item()
                labels[i, :answer_pos + 1] = -100
            else:
                # If no answer token found, mask entire sequence (shouldn't happen)
                labels[i, :] = -100
            
            # Also mask padding tokens
            labels[i][input_ids[i] == self.tokenizer.pad_token_id] = -100
        
        batch['labels'] = labels
        
        return batch


@dataclass
class DataCollatorForCLM:
    """
    Simple data collator for causal language modeling.
    Just creates labels from input_ids.
    """
    
    tokenizer: PreTrainedTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        # Labels are the same as input_ids for CLM
        labels = batch['input_ids'].clone()
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        batch['labels'] = labels
        
        return batch
