import os
import math
import numpy as np

# Define folder path where the documents are located
folder_path = 'C:/Users/Ashutosh Mehta/Documents/SEIR/Project3/25'

# Read documents from files
documents = []
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, 'r') as file:
            doc = ''.join([line.strip() for line in file])
            documents.append(doc)

# Extract document numbers and store in dictionary
document_dict = {}
for i, doc in enumerate(documents):
    docno_start_idx = doc.find("<DOCNO>") + 7
    docno_end_idx = doc.find("</DOCNO>")
    docno = doc[docno_start_idx:docno_end_idx]
    document_dict[i + 1] = docno

# Combine TITLE and TEXT content from documents
title_text_combined = []
for doc in documents:
    title_start_idx = doc.find("<TITLE>") + 7
    title_end_idx = doc.find("</TITLE>")
    title = doc[title_start_idx:title_end_idx]
    
    text_start_idx = doc.find("<TEXT>") + 6
    text_end_idx = doc.find("</TEXT>")
    text = doc[text_start_idx:text_end_idx]
    
    combined_text = (title + " " + text).lower()
    title_text_combined.append(combined_text)

# Custom tokenizer function to process text
def custom_tokenizer(text):
    tokens = []
    current_token = ''
    for char in text:
        if char.isalnum() or char == '_':
            current_token += char
        elif current_token:
            tokens.append(current_token)
            current_token = ''
    if current_token:
        tokens.append(current_token)
    return tokens

# Tokenize each document
tokenized_docs = [custom_tokenizer(doc) for doc in title_text_combined]

# Map each unique token to a unique ID
token_to_id = {}
token_id_counter = 1
for doc_tokens in tokenized_docs:
    for token in doc_tokens:
        if token not in token_to_id:
            token_to_id[token] = token_id_counter
            token_id_counter += 1

# Calculate IDF for each token
token_idf = {}
total_docs = len(tokenized_docs)
for token, token_id in token_to_id.items():
    df = sum(1 for doc_tokens in tokenized_docs if token in doc_tokens)
    idf = math.log(total_docs / df) if df > 0 else 0
    token_idf[token] = idf

# Create TF-IDF matrix
tfidf_matrix = []
for doc_tokens in tokenized_docs:
    tfidf = []
    for token, token_id in token_to_id.items():
        tf = doc_tokens.count(token)
        idf = token_idf[token]
        tfidf.append(tf * idf)
    tfidf_matrix.append(tfidf)

# Function to calculate cosine similarity
def cosine_similarity(vector1, vector2):
    dot_product = sum(w1 * w2 for w1, w2 in zip(vector1, vector2))
    norm_vector1 = math.sqrt(sum(w ** 2 for w in vector1))
    norm_vector2 = math.sqrt(sum(w ** 2 for w in vector2))
    return dot_product / (norm_vector1 * norm_vector2) if norm_vector1 and norm_vector2 else 0

# Calculate similarity matrix
similarity_matrix = np.zeros((total_docs, total_docs))
for i in range(total_docs):
    for j in range(i + 1, total_docs):
        similarity_matrix[i, j] = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
        similarity_matrix[j, i] = similarity_matrix[i, j]

# Store top similarity scores
top_similarity_scores = {}
for i in range(total_docs):
    for j in range(i + 1, total_docs):
        docno_i = document_dict[i + 1]
        docno_j = document_dict[j + 1]
        score = similarity_matrix[i, j]
        top_similarity_scores[(docno_i, docno_j)] = score

# Sort the similarity scores in descending order and extract top 50
top_50_similarity_scores = dict(sorted(top_similarity_scores.items(), key=lambda x: x[1], reverse=True)[:50])

# Display top 50 similarity scores
print("Top 50 similarity scores:")
print(top_50_similarity_scores)
