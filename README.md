-----------------------------------------------
SENTIMENT ANALYSIS WITH NLP & TRANSFORMERS
AI Academia Internship | RAJ JAT | October 2026

ABOUT
End-to-end sentiment analysis on the IMDB Movie Reviews dataset.
Compares classical ML, deep learning, and transformer-based approaches.

MODELS BUILT
1. Logistic Regression (TF-IDF baseline)
2. Naive Bayes (TF-IDF baseline)
3. Artificial Neural Network (PyTorch)
4. DistilBERT Transformer (Hugging Face)

DATASET
IMDB Movie Reviews — 50,000 reviews (25k train, 25k test)
Labels: Positive (1) and Negative (0)
Source: Hugging Face datasets library

KEY FINDINGS  
1. The Logistic Regression baseline achieved an 88.16% accuracy, while the DistilBERT transformer achieved 86.40% on a 500-sample subset. Transformers capture contextual relationships between words, whereas TF-IDF models do not. For production systems like automated review moderation, deploying the transformer model and evaluating it on a full dataset will provide a more robust understanding of its ability to reduce false positives/negatives compared to classical baselines.
2. Naive Bayes underperformed the Logistic Regression baseline, scoring 84.03%. Naive Bayes assumes all features (words) are independent, which is rarely true in natural language. If computational resources are strictly limited and deep learning cannot be used, Logistic Regression is the superior lightweight choice for this dataset.
3. The Logistic Regression model was highly effective given its simplicity and training speed. Classical TF-IDF vectorization with 5000 max features still captures the most heavily weighted sentiment-bearing words. It serves as an excellent fallback model or edge-device classifier where transformer hosting is too expensive.
4. The simple Feed-Forward ANN was restricted to a single training epoch for speed, achieving 88.52% accuracy. The network weights likely did not fully converge, meaning its performance does not represent the true ceiling of neural networks on this data. Developers should allocate proper GPU training time for multiple epochs to evaluate ANN architectures fairly against classical ML models.
5. DistilBERT operates natively on raw text rather than pre-processed TF-IDF arrays. The model handles its own tokenization and embeddings, preserving sequence data. Engineers can streamline data pipelines by removing manual punctuation stripping, lowercasing, and vectorization steps when deploying transformer APIs.

RESULTS SUMMARY
Logistic Regression: 88.16%
Naive Bayes        : 84.03%
ANN                : 88.52%
DistilBERT         : 86.40%

HOW TO RUN

->pip install pandas numpy matplotlib seaborn scikit-learn

->pip install transformers torch datasets

->Open Sentiment_Analysis_Final.ipynb in Jupyter and run all cells.

AUTHOR
RAJ JAT — AI Academia AI/ML Internship 2026
-----------------------------------------------
