# chunking_utils.py

def chunk_text(text, max_words=300):
    """
    Splits long text into word-based chunks.
    Used when prompt exceeds context window.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks
