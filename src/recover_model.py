import os
import shutil
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def recover():
    checkpoint_path = "results/checkpoint-352"
    final_path = "models/bert_classifier"
    
    print(f"Recovering model from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found!")
        return

    # 1. Load the model from the checkpoint
    print("Loading model weights...")
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)
    
    # 2. Load the standard tokenizer (since checkpoints might not have it)
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    
    # 3. Save both to the final destination
    print(f"Saving to {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("Success! Model recovered.")

if __name__ == "__main__":
    recover()