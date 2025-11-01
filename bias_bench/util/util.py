def _is_generative(model):
    # Checks if we are running an autoregressive model.
    generative_class = model in [
        "GPT2LMHeadModel",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPGPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel",
        "SelfDebiasGPT2LMHeadModel",
        "GPTNeoXForCausalLM",
        "QuantizedGPTNeoXForCausalLM",
        "SelfDebiasLLAMALMHeadModel",
        "SelfDebiasOPTLMHeadModel",
        "AutoModelForCausalLM"
    ]
    if not generative_class:
        for m in ['opt', 'llama', 'mistral', 'causal', 'qwen', 'gemma', 'gpt', 'bloom', 'granite']:
            if m in model.lower():
                generative_class=True
    return generative_class


def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return model in [
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
        "SelfDebiasLLAMALMHeadModel",
        "SelfDebiasOPTLMHeadModel",
    ]
