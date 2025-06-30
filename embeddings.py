from transformers import AutoTokenizer, AutoModel
import torch

def get_embeddings(model_name, sentence):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)

        # Choose which embedding to use (last hidden state, mean pooling, etc.)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over tokens

    return embeddings

models = [
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "roberta-base"
]

user_sentence = input("Enter a sentence to generate embeddings: ")

for model_name in models:
    print(f"\nModel: {model_name}")
    embeddings = get_embeddings(model_name, user_sentence)
    print("Embedding shape:", embeddings.shape)
    print("First 10 embedding values:", embeddings[0][:10].tolist())
    print("-" * 50)