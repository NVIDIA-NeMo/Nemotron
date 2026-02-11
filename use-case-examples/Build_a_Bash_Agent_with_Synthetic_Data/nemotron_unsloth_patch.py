"""
Monkey-patch for Nemotron models to work with Unsloth's efficient GRPO implementation.

Nemotron uses custom modeling code (modeling_nemotron_h.py) that doesn't respect
Unsloth's UNSLOTH_RETURN_HIDDEN_STATES environment variable. This patch fixes that.

Usage:
    from nemotron_unsloth_patch import patch_nemotron_for_unsloth_grpo
    
    model, tokenizer = FastLanguageModel.from_pretrained(...)
    patch_nemotron_for_unsloth_grpo(model)
    model = FastLanguageModel.get_peft_model(model, ...)
"""

import os
from typing import Optional


def patch_nemotron_for_unsloth_grpo(model):
    """
    Patch Nemotron's forward method to respect UNSLOTH_RETURN_HIDDEN_STATES.
    
    Unsloth's efficient GRPO sets UNSLOTH_RETURN_HIDDEN_STATES=1 expecting the model
    to return hidden states in the 'logits' field. Nemotron's custom modeling code
    doesn't support this - this patch adds that support.
    
    Args:
        model: The Nemotron model (can be wrapped in PEFT or not)
        
    Returns:
        None (modifies the model class in-place)
    """
    # Get the actual causal LM model (unwrap PEFT if needed)
    if hasattr(model, 'model'):
        base_model = model.model
        if hasattr(base_model, 'model'):
            causal_lm = base_model.model  # PeftModel -> base_model -> actual model
        else:
            causal_lm = base_model
    else:
        causal_lm = model
    
    # Check if this is a NemotronH model
    model_class_name = causal_lm.__class__.__name__
    if "NemotronH" not in model_class_name:
        print(f"Model is {model_class_name}, not NemotronH - skipping patch")
        return
    
    # Check if already patched
    if getattr(causal_lm.__class__, '_unsloth_patched', False):
        print(f"✓ {model_class_name} already patched for Unsloth GRPO")
        return
    
    # Store original forward
    original_forward = causal_lm.__class__.forward
    
    def patched_forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        cache_params=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_cache=None,
        cache_position=None,
        attention_mask=None,
        **kwargs,
    ):
        """Patched forward that respects UNSLOTH_RETURN_HIDDEN_STATES."""
        # Check if Unsloth wants hidden states instead of logits
        return_hidden_states_as_logits = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
        
        if not return_hidden_states_as_logits:
            # Normal forward pass
            return original_forward(
                self,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                cache_params=cache_params,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=use_cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Unsloth GRPO mode: return hidden states in the logits field
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get hidden states from backbone
        nemotron_h_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = nemotron_h_outputs[0]
        
        # Return hidden states in the logits field (this is what Unsloth GRPO expects)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=None,
            logits=hidden_states,  # Hidden states, not logits!
            past_key_values=None,
            hidden_states=nemotron_h_outputs.hidden_states if hasattr(nemotron_h_outputs, 'hidden_states') else None,
            attentions=nemotron_h_outputs.attentions if hasattr(nemotron_h_outputs, 'attentions') else None,
        )
    
    # Apply the patch
    causal_lm.__class__.forward = patched_forward
    causal_lm.__class__._unsloth_patched = True
    print(f"✓ Patched {model_class_name}.forward for Unsloth GRPO compatibility")


def verify_patch(model, tokenizer):
    """
    Verify that the monkey-patch is working correctly.
    
    Args:
        model: The patched model
        tokenizer: The tokenizer
        
    Returns:
        bool: True if patch is working, False otherwise
    """
    import torch
    
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    
    try:
        test_input = tokenizer("Test", return_tensors="pt").input_ids
        if torch.cuda.is_available():
            test_input = test_input.cuda()
        
        with torch.no_grad():
            test_output = model(test_input)
        
        # Hidden dim for Nemotron-Nano-9B is 4480
        # Vocab size is 131072
        # If patch is working, we should get hidden_dim (4480) not vocab_size
        is_working = test_output.logits.shape[-1] == 4480
        
        if is_working:
            print("✓ Monkey-patch verified: model returns hidden states when UNSLOTH_RETURN_HIDDEN_STATES=1")
        else:
            print(f"✗ WARNING: Patch may not be working. Got shape {test_output.logits.shape}, expected [..., 4480]")
        
        return is_working
    finally:
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
