# tokenizer_utils.py
import tiktoken

def count_tokens(text, model_name="gpt-3.5-turbo"):
    """
    Counts tokens in a given text.
    Using OpenAI tokenizer as approximation (industry practice).
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)
