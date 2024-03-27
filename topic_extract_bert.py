from bertopic import BERTopic
from transformers import GPT2Model, GPT2Tokenizer

def gpt_embedding(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2Model.from_pretrained('gpt2-medium')

def extract_topic_words(document):
    # Split the document into paragraphs
    paragraphs = document.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]  # Remove empty paragraphs
    
    topic_model = BERTopic(
        embedding_model=gpt_embedding,
        calculate_probabilities=True,
        verbose=True
    )
    
    topic_model.fit(paragraphs)
    
    topic_distr, _ = topic_model.approximate_distribution(paragraphs, use_embedding_model=True)
    
    all_topics = topic_model.get_topics()
    
    topic_words = []
    for topic_id in all_topics:
        # append the list of words and not their percentages
        words = [word for word, _ in all_topics[topic_id]]
        topic_words.append(words)
    
    return topic_words