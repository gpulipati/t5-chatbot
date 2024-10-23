# T5-Chatbot

## Overview
**T5-Chatbot** is a question-answering system built using the **T5-small** model, fine-tuned for generating answers based on input questions and relevant context. The project demonstrates how to utilize prompt engineering techniques to guide a transformer model for chatbot-like functionalities, suitable for tasks such as FAQ systems, customer support, and general-purpose knowledge extraction.

## Features
- Fine-tuned **T5-small** for question-answering tasks
- Uses web-scraped content as context for generating answers
- Supports flexible prompt engineering for various text generation use cases
- Lightweight, efficient, and deployable for API-based chatbot systems

## Project Structure
- `data/`: Contains datasets used for fine-tuning the model (if applicable).
- `scripts/`: Python scripts for fine-tuning, inference, and API setup.
- `model/`: Pre-trained and fine-tuned models, configuration files.
- `notebooks/`: Google Colab notebook with full model training pipeline.
- `README.md`: Project description and usage guide.

## Dependencies
The project relies on the following libraries:
- Python 3.x
- `transformers` (Hugging Face)
- `torch` (PyTorch)
- `datasets`
- `requests`, `beautifulsoup4` (for web scraping)
- `flask` or `fastapi` (optional for API deployment)

## How It Works
1. **Model**: The T5-small model is fine-tuned to take a question and context (extracted from a website or a specific dataset) as input and generate a relevant answer.
2. **Prompt Engineering**: Questions are formatted as: 
    ```
    question: {Your Question} context: {Your Context}
    ```
3. **Training**: The model is fine-tuned using curated question-answer pairs with relevant context extracted from a website (e.g., Wikipedia).
4. **Inference**: Once trained, the chatbot generates responses based on new user queries by utilizing the prompt-engineered input format.

## Usage
1. **Fine-tuning the model**:
   - Run the Google Colab notebook (`notebooks/t5_finetune.ipynb`) to fine-tune the model using a custom dataset or web content.
  
2. **Question Answering**:
   - Use the `generate_answer()` function in `scripts/prompt_engineer.py` to provide a question and context, and get the model’s response.

3. **Example Usage**:
    ```python
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    
    # Load pre-trained model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Question and context
    question = "What is machine learning?"
    context = "Machine learning is a field of study that gives computers the ability to learn from data."
    
    # Generate answer
    answer = generate_answer(question, context)
    print("Answer:", answer)
    ```

## Fine-tuning Instructions
To fine-tune the model on your own data:
1. Gather a dataset with question-answer pairs and their corresponding context.
2. Use the preprocessing function in the Colab notebook to tokenize and format your data.
3. Train the model using a GPU environment (e.g., Google Colab).
4. Save the fine-tuned model for later inference.

## Deployment (Optional)
You can deploy this model as an API using `Flask` or `FastAPI` to allow users to interact with the chatbot in real time. Here's a simple example of how to set up an API:

```bash
pip install flask
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    question = data.get("question")
    context = data.get("context")
    
    answer = generate_answer(question, context)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


## Future Improvements
1. Expand to more complex domains: Fine-tune on more diverse datasets for improved generalization.
2. Model Optimizations: Implement quantization or distillation to improve performance.
3. Interactive Features: Add multi-turn conversations or dialogue management for more sophisticated chatbot behavior.

## License
MIT License.
