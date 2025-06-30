from transformers import AutoTokenizer, AutoModel
import torch
import gensim.downloader as api
import numpy as np

def get_token_embeddings(model_name, sentence):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)

        # Get individual token embeddings (last hidden state)
        token_embeddings = outputs.last_hidden_state[0]  # Remove batch dimension
        
        # Get token IDs and convert to tokens
        token_ids = inputs['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    return tokens, token_ids, token_embeddings

def get_word_embeddings(model_name, sentence):
    try:
        # Load pre-trained model (this will download if not available)
        print(f"Loading {model_name}... (this may take a while on first run)")
        model = api.load(model_name)
        
        # Simple word tokenization (split by spaces and remove punctuation)
        words = sentence.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()
        
        embeddings = []
        tokens = []
        token_ids = []
        
        for i, word in enumerate(words):
            if word in model:
                embeddings.append(model[word])
                tokens.append(word)
                token_ids.append(i)  # Use index as token ID for word embeddings
            else:
                # Handle out-of-vocabulary words
                tokens.append(f"{word} (OOV)")
                embeddings.append(np.zeros(model.vector_size))
                token_ids.append(i)
        
        return tokens, token_ids, np.array(embeddings)
    
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Skipping this model...")
        return [], [], []

def display_token_table(model_name, tokens, token_ids, embeddings, is_numpy=False):
    print(f"\nModel: {model_name}")
    print(f"{'Token':<15} {'Token ID':<10} {'Embedding (first 5 values)'}")
    print("-" * 80)
    
    for token, token_id, embedding in zip(tokens, token_ids, embeddings):
        if is_numpy:
            embedding_str = str(embedding[:5].tolist())[1:-1]  # Remove brackets
        else:
            embedding_str = str(embedding[:5].tolist())[1:-1]  # Remove brackets
        print(f"{token:<15} {token_id:<10} [{embedding_str}...]")

# Transformer models
transformer_models = [
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "roberta-base"
]

user_sentence = input("Enter a sentence to generate embeddings: ")

print("\n" + "="*80)
print("TRANSFORMER MODELS (Contextual Embeddings)")
print("="*80)

for model_name in transformer_models:
    tokens, token_ids, embeddings = get_token_embeddings(model_name, user_sentence)
    display_token_table(model_name, tokens, token_ids, embeddings)
    print("=" * 80)

# Optional: Try GloVe if available
print("\n" + "="*80)
print("WORD EMBEDDING MODELS (Static Embeddings)")
print("="*80)

try:
    tokens, token_ids, embeddings = get_word_embeddings("glove-wiki-gigaword-100", user_sentence)
    if len(tokens) > 0:
        display_token_table("glove-wiki-gigaword-100", tokens, token_ids, embeddings, is_numpy=True)
        print("=" * 80)
except:
    print("GloVe model unavailable (requires internet connection on first use)")
    print("=" * 80)