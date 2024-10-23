import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
from train import tokenize
# Initialize
app = Flask(__name__)

# Load the pretrained models
model = BertForSequenceClassification.from_pretrained('models/classfier') 
tokenizer = BertTokenizer.from_pretrained('models/tokenizer')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

def classification(review):
    inputs = tokenize(review, tokenizer)

    # Move the inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiment = 'positive' if prediction == 1 else 'negative'
    return sentiment

# Define the POST endpoint to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json

    # Check data intergraty 
    if 'review' not in data or not data['review'].strip():
        return jsonify({'error': 'Invalid input. Please provide a valid review.'}), 400
    review = data['review']
    
    # Inference
    sentiment = classification(review)
    return jsonify({'sentiment': sentiment})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
