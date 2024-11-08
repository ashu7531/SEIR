# Document Similarity Analysis with TF-IDF and Cosine Similarity

This project analyzes similarity between documents using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **cosine similarity**. Given a collection of documents, it calculates TF-IDF vectors to represent the importance of terms within each document, then computes cosine similarity scores to measure how closely related each document pair is.

## Project Overview

1. **Data Processing**: The script reads documents from a specified folder, extracting relevant text sections for analysis.
2. **Tokenization**: Custom tokenization is applied to break down the text into meaningful alphanumeric tokens.
3. **TF-IDF Calculation**: Each document's content is converted into a TF-IDF vector to quantify the importance of each word relative to all documents.
4. **Cosine Similarity Computation**: Cosine similarity scores are calculated for each document pair, producing a matrix of similarity scores.
5. **Top Similar Documents**: The project outputs the top 50 most similar document pairs, sorted by their similarity score.

## Output

The output is a ranked list of the 50 document pairs with the highest similarity scores. This provides insight into which documents share the most content or thematic similarity, based on the TF-IDF representation of their text.

## Example Output

The output format is a dictionary showing pairs of document IDs along with their cosine similarity score:
```python
Top 50 similarity scores:
{
    ('DOC1', 'DOC2'): 0.85,
    ('DOC3', 'DOC5'): 0.78,
    ...
}
