import torch
from bertDataset import BertDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
def tokenize(texts,tokenizer):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv("Dataset\Restaurant_Reviews.tsv",sep='\t')
    df['Review'] = df['Review'].astype(str)
    df['Liked'] = df['Liked'].astype(int)

    # Train, Validation Seperation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Load model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize text
    
    train_encodings = tokenize(train_df['Review'].tolist(),tokenizer)
    val_encodings = tokenize(val_df['Review'].tolist(),tokenizer)

    # Labels
    train_labels = train_df['Liked'].values
    val_labels = val_df['Liked'].values

    # Parameters
    num_labels = len(df['Liked'].unique())
    epoch = 30
    lr = 1E-5
    batch_size =32
    train_shuffle = True
    test_shuffle = False

    # Create datasets and optimizer
    train_dataset = BertDataset(train_encodings, train_labels)
    val_dataset = BertDataset(val_encodings, val_labels)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Load data 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=test_shuffle)


    writer = SummaryWriter(log_dir='runs/bert_sentiment_analysis')
    best_val_accuracy = 0
    best_model = None
    # Training loop
    for i in range(epoch):  # Number of epochs
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss # Cross entropy 
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {i+1}, Training Loss: {avg_train_loss}")
        writer.add_scalar('Loss/train', avg_train_loss, i+1)
        # Validation
        model.eval()
        correct, total = 0, 0
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask,labels=labels)
                predictions = outputs.logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            print(f"Epoch {i+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")
            writer.add_scalar('Loss/validation', avg_val_loss, i+1)
            writer.add_scalar('Accuracy/validation', val_accuracy, i+1)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model
        # Save the model
    best_model.save_pretrained('models/classfier')
    tokenizer.save_pretrained('models/tokenizer')
    writer.close()