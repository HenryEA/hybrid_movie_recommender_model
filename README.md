# Hybrid Movie Recommender System

This repository contains the code for a hybrid movie recommendation system that combines collaborative filtering, content-based filtering, sequential learning and popularity scores to recommend movies based on user preferences.

## Project Overview

The goal of this system is to recommend movies that users may enjoy based on their previous ratings, movie interests and the order of their interactions. It also supports recommendations for new users based on their preferred genres and selected movies.

## Key Components

### 1. Data Preparation

- Dataset: MovieLens 100K, containing 100,000 ratings from 943 users across 1,682 movies.
- Chronological Split: Each user's latest rating is reserved for testing, the second latest for validation and the remaining ratings for training.
- Positive Interactions: Ratings of 4 or 5 are used to identify movies a user liked.

### 2. Model Architecture

- Matrix Factorization: Learns user and movie preferences using embeddings, with user and movie biases.
- Neural Collaborative Filtering: Uses a neural network to learn nonlinear relationships between users and movies.
- Content Model: Uses the pretrained Hugging Face `all-MiniLM-L6-v2` model to represent movie titles and genres as embeddings.
- Sequential Transformer: Learns from the order of positively rated movies to predict the next movie a user may like.
- Popularity Model: Combines average ratings and rating counts to score movies.

### 3. Hybrid Recommendations

Model scores are converted to percentile ranks and combined using the following weights:

- Matrix Factorization: 35%
- Neural Collaborative Filtering: 25%
- Sequential Transformer: 20%
- Content Model: 15%
- Popularity Model: 5%

Movies already rated by the user are excluded from the final recommendations.

### 4. Additional Features

- Finds movies with similar titles and genres using content embeddings.
- Recommends movies to new users using preferred genres, liked movies and popularity.
- Provides simple explanations based on genres shared with previously liked movies.
- Saves trained models, embeddings and evaluation results for reuse.

## Technical Stack

Python, PyTorch, Pandas, NumPy, Scikit-learn, Sentence Transformers, Matplotlib, Seaborn and Joblib.

## Model Evaluation

The models are evaluated using top-10 recommendations. Testing includes 486 users whose held-out movie received a rating of 4 or 5.

Evaluation metrics include Hit Rate@10, Precision@10, Recall@10, NDCG@10, MRR@10 and catalogue coverage.

## Results

The recorded notebook results show that the hybrid model achieved the highest Hit Rate@10 and NDCG@10 among the evaluated models.

| Model | Hit Rate@10 | NDCG@10 |
|---|---:|---:|
| Hybrid Model | 6.58% | 0.0330 |
| Popularity | 5.56% | 0.0242 |
| Matrix Factorization | 4.12% | 0.0157 |
| Neural Collaborative Filtering | 3.50% | 0.0141 |
| Sequential Transformer | 2.67% | 0.0160 |
| Content Model | 2.26% | 0.0095 |

The hybrid model placed the held-out liked movie among its top 10 recommendations for 32 of the 486 evaluated users.
