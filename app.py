import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import gradio as gr

# Load dataset
file_path = 'mining_rules_regulations_dataset_10000.csv'  # Upload this file to your Space
df = pd.read_csv(file_path)

# Extract User Input and Bot Response columns
user_inputs = df['User Input'].tolist()
bot_responses = df['Bot Response'].tolist()

# Load the pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all user inputs in the dataset into embeddings
user_input_embeddings = model.encode(user_inputs, convert_to_tensor=True)

# Function to get the chatbot response
def get_bot_response(user_query):
    user_query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_query_embedding, user_input_embeddings)
    closest_match_index = torch.argmax(similarities).item()
    threshold = 0.5
    max_similarity = similarities[0][closest_match_index].item()

    if max_similarity < threshold:
        return "I'm sorry, but I can only answer questions related to mining regulations."
    return bot_responses[closest_match_index]

# Gradio Interface
def chatbot_interface(user_query):
    return get_bot_response(user_query)

interface = gr.Interface(fn=chatbot_interface,
                         inputs="text",
                         outputs="text",
                         title="Mining Regulations Chatbot",
                         description="Ask me questions about rules and regulations in the mining industry!")

interface.launch()