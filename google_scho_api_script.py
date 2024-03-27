import logging
import time
from collections import Counter

import nltk
import requests
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tenacity import retry, stop_after_attempt, wait_fixed

from topic_extract_gpt_api import extract_topics_gpt

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

API_ENDPOINT = "https://serpapi.com/search.json"
API_KEY = ""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_key_terms(file_content):
    words = simple_preprocess(str(file_content), deacc=True)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    key_terms = [lemmatizer.lemmatize(word) for word in words]
    return key_terms


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_research_papers(key_terms):
    if key_terms is None or len(key_terms) == 0:
        return []
    
    query = " ".join(key_terms)
    params = {
        "q": query,
        "api_key": API_KEY,
        "engine": "google_scholar",
        "num": 10
    }

    try:
        response = requests.get(API_ENDPOINT, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return data.get("organic_results", [])
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred during API request: {e}")
        return []


def calculate_relevance_score(paper, key_terms):
    score = sum(term in paper["title"].lower() or term in paper["snippet"].lower() for term in key_terms)
    score += paper.get("cited_by", {}).get("total", 0) * 0.1
    return score


def get_merged_terms_papers(document):
    current_key_terms = extract_key_terms(document)
    current_key_term_counts = Counter(current_key_terms)
    top_key_terms = [term for term, _ in current_key_term_counts.most_common(10)]

    logging.info(f"Top Key Terms: {top_key_terms}")

    merged_terms_papers = get_research_papers(top_key_terms)
    for paper in merged_terms_papers:
        paper["relevance_score"] = calculate_relevance_score(paper, top_key_terms)

    merged_terms_papers.sort(key=lambda x: x["relevance_score"], reverse=True)

    return merged_terms_papers


def get_papers_by_topic(document):
    topics = extract_topics_gpt(document)
    papers_by_topic = []

    for topic in topics:
        words = simple_preprocess(str(topic), deacc=True)
        topic_papers = get_research_papers(words)

        for paper in topic_papers:
            paper["relevance_score"] = calculate_relevance_score(paper, words)

        topic_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
        papers_by_topic.append(topic_papers)

    return papers_by_topic


def suggest_research_papers(file_path):
    old_content = ""

    while True:
        with open(file_path, "r") as file:
            current_content = file.read()

        if current_content == old_content:
            time.sleep(1)
            continue

        papers_by_topic = get_papers_by_topic(current_content)
        merged_terms_papers = get_merged_terms_papers(current_content)

        num_suggestions = 5
        top_merged_terms_papers = merged_terms_papers[:num_suggestions]
        top_papers_by_topic = papers_by_topic[:num_suggestions]

        logging.info("Suggested Research Papers for top terms:")
        for paper in top_merged_terms_papers:
            logging.info(f"Title: {paper['title']}")
            logging.info("====================================")

        logging.info("Suggested Research Papers by found topics:")
        for papers in top_papers_by_topic:
            for paper in papers:
                logging.info(f"Title: {paper['title']}")
                logging.info("====================================")

        old_content = current_content
        time.sleep(1)


if __name__ == "__main__":
    file_path = "test_document.txt"
    suggest_research_papers(file_path)