import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pprint import pprint

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
    
    # Remove stopwords and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Remove additional domain-specific stopwords
    domain_stop_words = ['introduction', 'conclusion', 'title', 'document', 'section', 'topic', 'model', 'method', 'approach', 'technique']
    words = [word for word in words if word not in domain_stop_words]
    
    return ' '.join(words)

# Read a document from a file
file_path = 'test_document.txt'
document = read_document(file_path)

# Preprocess the document
processed_doc = preprocess_document(document)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words)

# Fit and transform the document
tfidf_matrix = vectorizer.fit_transform([processed_doc])

if __name__ == '__main__':
    # Set the number of topics (latent dimensions)
    num_topics = 5

    # Create and fit the LSA model
    lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa_model.fit(tfidf_matrix)

    # Get the topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lsa_model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append((topic_idx, top_features))

    # Print the discovered topics
    print("\nDiscovered Topics:")
    for topic_idx, top_features in topics:
        print(f"Topic {topic_idx}: {', '.join(top_features)}")

    # Get the topic distribution for the document
    topic_distribution = lsa_model.transform(tfidf_matrix)

    # Print the topic distribution for the document
    print("\nDocument Topic Distribution:")
    for topic_idx, prob in enumerate(topic_distribution[0]):
        print(f"Topic {topic_idx}: {prob}")
        print()