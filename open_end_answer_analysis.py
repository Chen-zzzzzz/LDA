import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download necessary NLTK resources (run once)
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

def analyze_open_ended_responses(responses, num_clusters=3):
    """
    Analyzes open-ended survey responses by performing:
    - Sentiment analysis,
    - Topic modeling (with optimal topic number selection via coherence scores),
    - Clustering using TF-IDF vectorization and K-Means,
    - Theme extraction based on LDA topic keywords.
    """
    # Sentiment Analysis using VADER
    analyzer = SentimentIntensityAnalyzer()
    sentiment_results = [analyzer.polarity_scores(response) for response in responses]
    sentiment_df = pd.DataFrame(sentiment_results)
    sentiment_df['response'] = responses

    # Text Preprocessing: Tokenization, stopword removal, and lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    custom_stopwords = {"would", "like", "wish", "think", "feel"}

    def preprocess(text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens
                  if word.isalpha() and word not in stop_words and word not in custom_stopwords]
        return tokens

    processed_responses = [preprocess(response) for response in responses]

    # Create Gensim Dictionary and Corpus for LDA
    dictionary = Dictionary(processed_responses)
    corpus = [dictionary.doc2bow(text) for text in processed_responses]

    # Function to compute coherence values for a range of topic numbers
    def compute_coherence_values(corpus, dictionary, k_values):
        coherence_values = []
        for k in k_values:
            lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=42)
            coherence_model = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherence_values.append(coherence_model.get_coherence())
        return coherence_values

    k_values = range(2, 10)
    coherence_values = compute_coherence_values(corpus, dictionary, k_values)
    optimal_k = k_values[np.argmax(coherence_values)]
    print(f"Optimal number of topics: {optimal_k}")

    # Train LDA model with the optimal number of topics
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_k, random_state=42)
    topics = lda_model.show_topics(num_words=10, formatted=False)
    topic_keywords = {topic_id: [word for word, _ in words] for topic_id, words in topics}

    # Generate topic distribution for each document
    topic_results = []
    for doc in corpus:
        topic_dist = lda_model[doc]
        topic_probs = [0.0] * optimal_k
        for topic_id, prob in topic_dist:
            topic_probs[topic_id] = prob
        topic_results.append(topic_probs)
    topic_df = pd.DataFrame(topic_results, columns=[f"Topic {i}" for i in range(optimal_k)])
    topic_df['response'] = responses

    # TF-IDF Vectorization for Clustering
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    X = vectorizer.fit_transform([" ".join(text) for text in processed_responses])
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    cluster_df = pd.DataFrame({'response': responses, 'cluster': clusters})
    silhouette_avg = silhouette_score(X, clusters)
    print(f"Silhouette Score: {silhouette_avg}")

    # Define themes based on LDA topic keywords
    themes = {f"Topic {topic_idx}": keywords for topic_idx, keywords in topic_keywords.items()}
    theme_counts = Counter()
    for response in responses:
        response_lower = response.lower()
        for theme, keywords in themes.items():
            if any(keyword in response_lower for keyword in keywords):
                theme_counts[theme] += 1

    # Return all analysis results in a dictionary
    return {
        'sentiment_results': sentiment_df,
        'topic_results': topic_df,
        'topic_keywords': topic_keywords,
        'cluster_results': cluster_df,
        'theme_counts': theme_counts,
    }

# File Reading and Analysis
filepath = "data/survey.txt"
try:
    with open(filepath, 'r', encoding='utf-8') as file:
        responses = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"Error: File not found at {filepath}")
    responses = []

if responses:
    analysis_results = analyze_open_ended_responses(responses)
    if analysis_results:
        # Print sentiment analysis results
        print("Sentiment Analysis:")
        print(analysis_results['sentiment_results'])
        
        # Print topic probabilities
        print("\nTopic Probabilities:")
        print(analysis_results['topic_results'])
        
        # Print topic keywords
        print("\nTopic Keywords:")
        for topic_id, keywords in analysis_results['topic_keywords'].items():
            print(f"Topic {topic_id}: {', '.join(keywords)}")
        
        # Print cluster assignments
        print("\nCluster Assignments:")
        print(analysis_results['cluster_results'])
        
        # Print theme counts
        print("\nTheme Counts:")
        print(analysis_results['theme_counts'])
        
        # Print average sentiment scores
        print("\nAverage Sentiment Scores:")
        print(analysis_results['sentiment_results'][['neg', 'neu', 'pos', 'compound']].mean())
        
        # Print most common themes
        print("\nMost Common Themes:")
        print(analysis_results['theme_counts'].most_common())
        
        # Print most negative responses
        print("\nMost Negative Responses:")
        print(analysis_results['sentiment_results'].sort_values(by='compound').head(3)[['response', 'compound']])
