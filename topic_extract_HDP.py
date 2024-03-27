import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import HdpModel
from gensim.corpora import Dictionary
from pprint import pprint
import string

# Download the stopwords and WordNet lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')  # Download the punkt tokenizer

# Define the stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to read a document from a file
def read_document(file_path):
    with open(file_path, 'r') as file:
        document = file.read()
    return document

def preprocess_document(document):
    # Tokenize the document
    words = nltk.word_tokenize(document.lower())
    
    # Remove punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
    
    # Remove stopwords and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Remove additional domain-specific stopwords
    domain_stop_words = ['introduction', 'conclusion', 'title', 'document', 'section', 'topic', 'model', 'method', 'approach', 'technique']
    words = [word for word in words if word not in domain_stop_words]
    
    # Remove single-character words
    words = [word for word in words if len(word) > 1]
    
    return words

# Read a document from a file
file_path = 'test_document.txt'
document = read_document(file_path)

# Preprocess the document
processed_doc = preprocess_document(document)

# Create a dictionary from the preprocessed document
dictionary = Dictionary([processed_doc])

# Create a corpus
corpus = [dictionary.doc2bow(processed_doc)]

# Train the HDP model
hdp_model = HdpModel(corpus, dictionary)

# Get the topics
topics = hdp_model.show_topics(formatted=False)

# Post-process the topics by removing non-informative terms
processed_topics = []
for topic_idx, topic in enumerate(topics):
    top_features = [word for word, _ in topic[1] if word not in stop_words and len(word) > 1]
    processed_topics.append((topic_idx, top_features))

# Print the discovered topics
print("\nDiscovered Topics:")
for topic_idx, top_features in processed_topics:
    print(f"Topic {topic_idx}: {', '.join(top_features)}")

# Get the topic distribution for the document
topic_distribution = hdp_model[corpus[0]]

# Print the topic distribution for the document
print("\nDocument Topic Distribution:")
for topic_idx, prob in topic_distribution:
    print(f"Topic {topic_idx}: {prob:.4f}")
    print()