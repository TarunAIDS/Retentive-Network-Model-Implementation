from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from data_utils import load_dataset, preprocess_data
from examples.example import hyperparameters

class DialogDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        inputs = self.tokenizer(sample['text'], return_tensors="pt", padding=True, truncation=True)
        inputs['labels'] = torch.tensor(sample['label'])
        return inputs


def custom_collate(batch):
    # Find the maximum sequence length in the batch
    max_len = max(len(item['input_ids'][0]) for item in batch)
    
    # Pad or truncate all sequences to the maximum length
    for item in batch:
        item['input_ids'] = torch.cat([item['input_ids'][0][:max_len], torch.zeros(max_len - len(item['input_ids'][0]), dtype=torch.long)]).unsqueeze(0)
        item['attention_mask'] = torch.cat([item['attention_mask'][0][:max_len], torch.zeros(max_len - len(item['attention_mask'][0]), dtype=torch.long)]).unsqueeze(0)
    
    # Stack the tensors
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch])
    }


# Load and preprocess data
dialogues = load_dataset("data/human_chat.txt")
processed_data = preprocess_data(dialogues)
train_data, val_data = train_test_split(processed_data, test_size=0.2)

# Use a tokenizer directly
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", force_download=True)

# Create dataset and dataloaders
train_dataset = DialogDataset(train_data, tokenizer)
val_dataset = DialogDataset(val_data, tokenizer)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate)

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, force_download=True)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["learning_rate"])

# Training loop
for epoch in range(hyperparameters["num_epochs"]):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        
        inputs = batch
        labels = inputs.pop("labels", None)  # Remove the 'labels' key from inputs
        
        outputs = model(**inputs)
        
        if labels is not None:
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch
            labels = inputs.pop("labels", None)  # Remove the 'labels' key from inputs
            
            outputs = model(**inputs)
            
            if labels is not None:
                val_loss += criterion(outputs.logits, labels)
        
        if len(val_loader) > 0:
            val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}/{hyperparameters["num_epochs"]}, Loss: {val_loss.item()}')

# Interactive mode
def interact_with_model(model, tokenizer):
    user_input = input("User: ")
    while user_input.lower() != 'exit':
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        predicted_labels = torch.argmax(outputs.logits, dim=1)
        response = "Positive" if predicted_labels == 1 else "Negative"
        print("Bot:", response)
        user_input = input("User: ")

# Start interactive mode
interact_with_model(model, tokenizer)