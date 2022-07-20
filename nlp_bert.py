import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize

from sentence_transformers import SentenceTransformer

def process_bert_similarity(base_document,documents):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentences = sent_tokenize(base_document)
    base_embeddings_sentences = model.encode(sentences)
    base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)
    vectors = []
    for i, document in enumerate(documents):
        sentences = sent_tokenize(document)
        embeddings_sentences = model.encode(sentences)
        embeddings = np.mean(np.array(embeddings_sentences), axis=0)
        vectors.append(embeddings)
#         print("making vector at index:", i)
    scores = cosine_similarity([base_embeddings], vectors).flatten()
    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(scores):
        if highest_score < score:
            highest_score = score
            highest_score_index = i
            most_similar_document = documents[highest_score_index]
    print("Most similar document by BERT with the score:",  highest_score)

base_document = "This is an example sentence for the document to be compared"
comparison_document = ["This is the collection of documents to be compared against the base_document"]

process_bert_similarity(base_document,comparison_document)
print ("Base document:", base_document)
print ("Comparison document:", comparison_document)

""" import string
import nltk

#nltk.download('all')
nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()



def preprocess(text):
    lowered = str.lower(text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(lowered)
    words = []
    for w in word_tokens:
        if w not in stop_words:
            if w not in string.punctuation:
                if len(w) > 1:
                    lemmatized = lemmatizer.lemmatize(w)
                    words.append(lemmatized)
    return words

def calculate_jaccard(word_tokens1, word_tokens2):
    both_tokens = word_tokens1 + word_tokens2
    union = set(both_tokens)
    intersection = set()
    for w in word_tokens1:
        if w in word_tokens2:
            intersection.add(w)
    jaccard_score = len(intersection)/len(union)
    return jaccard_score

def process_jaccard_similarity(base_document,documents):
    base_tokens = preprocess(base_document)
    all_tokens = []
    for i, document in enumerate(documents):
        tokens = preprocess(document)
        all_tokens.append(tokens)
        print("making word tokens at index:", i)
    all_scores = []
    for tokens in all_tokens:
        score = calculate_jaccard(base_tokens, tokens)
        all_scores.append(score)
        
    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(all_scores):
        if highest_score < score:
            highest_score = score
            highest_score_index = i
    most_similar_document = documents[highest_score_index]
    print("Most similar document by Jaccard with the score:", most_similar_document, highest_score)

base_document = "This is an example sentence for the document to be compared"
documents = ["This is the collection of documents to be compared against the base_document"]

process_jaccard_similarity(base_document,documents)
 """
""" from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

def preprocess(text):
    lowered = str.lower(text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(lowered)
    words = []
    for w in word_tokens:
        if w not in stop_words:
            if w not in string.punctuation:
                if len(w) > 1:
                    lemmatized = lemmatizer.lemmatize(w)
                    words.append(lemmatized)
    return words

def process_doc2vec_similarity(base_document,documents):
#     filename = './models/apnews_dbow/doc2vec.bin'
    filename = 'enwiki_dbow/doc2vec.bin'
    model= Doc2Vec.load(filename)
    tokens = preprocess(base_document)
    tokens = list(filter(lambda x: x in model.wv.vocab.keys(), tokens))
    base_vector = model.infer_vector(tokens)
    vectors = []
    for i, document in enumerate(documents):
        tokens = preprocess(document)
        tokens = list(filter(lambda x: x in model.wv.vocab.keys(), tokens))
        vector = model.infer_vector(tokens)
        vectors.append(vector)
        print("making vector at index:", i)
    scores = cosine_similarity([base_vector], vectors).flatten()
    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(scores):
        if highest_score < score:
            highest_score = score
            highest_score_index = i
    most_similar_document = documents[highest_score_index]
    print("Most similar document by Doc2vec with the score:", most_similar_document, highest_score)

base_document = "This is an example sentence for the document to be compared"
documents = ["This is the collection of documents to be compared against the base_document"]

process_doc2vec_similarity(base_document,documents)
 """
"""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def process_tfidf_similarity(base_document,documents):
    vectorizer = TfidfVectorizer()
    documents.insert(0, base_document)
    embeddings = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(cosine_similarities):
        if highest_score < score:
            highest_score = score
            highest_score_index = i
    most_similar_document = documents[highest_score_index]
    print("Most similar document by TF-IDF with the score:", most_similar_document, highest_score)
base_document = "This is an example sentence for the document to be compared"
documents = ["This is the collection of documents to be compared against the base_document"]

process_tfidf_similarity(base_document,documents)

from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
import tensorflow_hub as hub

def process_use_similarity(base_document,documents):
    filename = 'https://tfhub.dev/google/universal-sentence-encoder/4'
    model = hub.load(filename)
    base_embeddings = model([base_document])
    embeddings = model(documents)
    scores = cosine_similarity(base_embeddings, embeddings).flatten()
    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(scores):
        if highest_score < score:
            highest_score = score
            highest_score_index = i
    most_similar_document = documents[highest_score_index]
    print("Most similar document by USE with the score:", most_similar_document, highest_score)

base_document = "This is an example sentence for the document to be compared"
documents = ["This is the collection of documents to be compared against the base_document"]

process_use_similarity(base_document,documents)"""