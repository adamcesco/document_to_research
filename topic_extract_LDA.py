import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pprint import pprint
import logging
from gensim.models import CoherenceModel

# Set up logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Download the stopwords and WordNet lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

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
    words = simple_preprocess(str(document), deacc=True)
    
    # Remove stopwords and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Remove additional domain-specific stopwords
    domain_stop_words = ['introduction', 'conclusion', 'title', 'document', 'section']
    words = [word for word in words if word not in domain_stop_words]
    
    return words

# Read a document from a file
file_path = 'test_document.txt'
document = read_document(file_path)

# Preprocess the document
processed_doc = preprocess_document(document)

# Create a dictionary and corpus
dictionary = gensim.corpora.Dictionary([processed_doc])
corpus = [dictionary.doc2bow(processed_doc)]

# Set the number of topics and other parameters
num_topics = 5
alpha = 0.1  # Decreased alpha value to promote sparser topic distributions

# Train the LDA model
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha=alpha, random_state=42)


# Print the discovered topics
print("\nDiscovered Topics:")
pprint(lda_model.print_topics())

# Get the topic distribution for the document
doc_lda = lda_model[corpus[0]]

# Print the topic distribution for the document
print("\nDocument Topic Distribution:")
for topic_id, prob in doc_lda:
    print(f"Topic {topic_id}: {prob}")
    
    # Print the top words for each topic
    topic_words = lda_model.show_topic(topic_id, topn=10)
    print(f"Top Words: {', '.join([word for word, _ in topic_words])}")
    print()