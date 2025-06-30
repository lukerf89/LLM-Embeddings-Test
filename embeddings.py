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
        
        # Convert input IDs back to tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    return tokens, token_embeddings

def get_word_embeddings(model_name, sentence):
    try:
        # Load pre-trained model (this will download if not available)
        print(f"Loading {model_name}... (this may take a while on first run)")
        model = api.load(model_name)
        
        # Simple word tokenization (split by spaces and remove punctuation)
        words = sentence.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()
        
        embeddings = []
        tokens = []
        
        for word in words:
            if word in model:
                embeddings.append(model[word])
                tokens.append(word)
            else:
                # Handle out-of-vocabulary words
                tokens.append(f"{word} (OOV)")
                embeddings.append(np.zeros(model.vector_size))
        
        return tokens, np.array(embeddings)
    
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Skipping this model...")
        return [], []

def display_token_table(model_name, tokens, embeddings, is_numpy=False):
    print(f"\nModel: {model_name}")
    print(f"{'Token':<15} {'Embedding (first 5 values)'}")
    print("-" * 65)
    
    for token, embedding in zip(tokens, embeddings):
        if is_numpy:
            embedding_str = str(embedding[:5].tolist())[1:-1]  # Remove brackets
        else:
            embedding_str = str(embedding[:5].tolist())[1:-1]  # Remove brackets
        print(f"{token:<15} [{embedding_str}...]")

# Transformer models
transformer_models = [
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "roberta-base"
]

# Word embedding models
word_models = [
    "glove-wiki-gigaword-100"
]

user_sentence = input("Enter a sentence to generate embeddings: ")

print("\n" + "="*65)
print("TRANSFORMER MODELS (Contextual Embeddings)")
print("="*65)

for model_name in transformer_models:
    tokens, embeddings = get_token_embeddings(model_name, user_sentence)
    display_token_table(model_name, tokens, embeddings)
    print("=" * 65)

print("\n" + "="*65)
print("WORD EMBEDDING MODELS (Static Embeddings)")
print("="*65)

for model_name in word_models:
    tokens, embeddings = get_word_embeddings(model_name, user_sentence)
    if len(tokens) > 0:  # Only display if model loaded successfully
        display_token_table(model_name, tokens, embeddings, is_numpy=True)
        print("=" * 65)