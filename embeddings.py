from transformers import AutoTokenizer, AutoModel
import torch

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

def display_token_table(model_name, tokens, embeddings):
    print(f"\nModel: {model_name}")
    print(f"{'Token':<15} {'Embedding (first 5 values)'}")
    print("-" * 65)
    
    for token, embedding in zip(tokens, embeddings):
        embedding_str = str(embedding[:5].tolist())[1:-1]  # Remove brackets
        print(f"{token:<15} [{embedding_str}...]")

models = [
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "roberta-base"
]

user_sentence = input("Enter a sentence to generate embeddings: ")

for model_name in models:
    tokens, embeddings = get_token_embeddings(model_name, user_sentence)
    display_token_table(model_name, tokens, embeddings)
    print("=" * 65)