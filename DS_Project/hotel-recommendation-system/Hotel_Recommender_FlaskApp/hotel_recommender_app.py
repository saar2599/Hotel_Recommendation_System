from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import string

app = Flask(__name__)

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the KMeans object globally
kmeans = None

# Function to clean text
def clean_text(text):
    return text.replace(r"\/", "").translate(str.maketrans('', '', string.punctuation)).replace(r"\d+", "").replace(r"\s{2,}", " ").lower()

# Load and preprocess the dataset
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['cleaned_reviews'] = data['Review'].apply(clean_text)
    return data

final_reviews = load_and_preprocess_data('data/final_tripadvisor.csv')

# Encode reviews and fit KMeans
def generate_embeddings_and_fit_kmeans(reviews):
    global kmeans
    embeddings = model.encode(reviews.tolist(), batch_size=64, show_progress_bar=True, convert_to_tensor=False)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(embeddings)
    return embeddings, kmeans.labels_

embeddings, cluster_labels = generate_embeddings_and_fit_kmeans(final_reviews['cleaned_reviews'])
final_reviews['cluster'] = cluster_labels  # Store cluster assignments
final_reviews['embeddings'] = list(embeddings)  # Store embeddings

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_review = request.form['review']
    cleaned_review = clean_text(user_review)  # Clean the user review
    new_review_embedding = model.encode([cleaned_review], convert_to_tensor=False)
    new_review_cluster = kmeans.predict(new_review_embedding.reshape(1, -1))[0]
    clustered_reviews = final_reviews[final_reviews['cluster'] == new_review_cluster]

    top_hotels = {}
    for hotel, group in clustered_reviews.groupby('Hotel Name'):
        hotel_review_embeddings = np.vstack(group['embeddings'])
        similarities = util.pytorch_cos_sim(new_review_embedding, hotel_review_embeddings).flatten()

        # Find the index of the review with the highest similarity score
        best_match_index = similarities.argmax().item()
        print("Index type:", type(best_match_index))
        print("Index value:", best_match_index)
        best_match_review = group.iloc[best_match_index]['Review']
        best_match_score = round(similarities[best_match_index].item(), 2)

        # Store the best match review and its score
        top_hotels[hotel] = (best_match_score, best_match_review)

    # Sort hotels based on the highest similarity score and select the top 3
    recommended_hotels = sorted(top_hotels.items(), key=lambda x: x[1][0], reverse=True)[:3]
    return render_template('result.html', hotels=recommended_hotels, user_review = cleaned_review)

if __name__ == '__main__':
    app.run(debug=True)
