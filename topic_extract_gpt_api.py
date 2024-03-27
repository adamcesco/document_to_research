import logging
from typing import List

from openai import OpenAI

API_KEY = ""

client = OpenAI(api_key=API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_topics_gpt(document: str) -> List[str]:
    """
    Extract topics from a document using the OpenAI GPT-3.5 model.

    Args:
        document (str): The document text to extract topics from.

    Returns:
        List[str]: A list of extracted topics.
    """
    try:
        prompt = (
            f"{document}\n\n"
            "Use the above document as a reference. Please suggest queries to Google Scholar API "
            "that would return interesting research papers on all the in-depth topics and items "
            "mentioned in the document. I am going to use Python to split your response by newlines "
            "and send each line as a separate query."
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI tool that will assist me in extracting topics "
                               "from a document to be used to query for interesting research papers.",
                },
                {"role": "user", "content": prompt},
            ],
            model="gpt-3.5-turbo",
        )

        topics = response.choices[0].message.content.split("\n")
        return topics

    except Exception as e:
        logger.exception(f"Error occurred during topic extraction: {str(e)}")
        return []


def read_document(file_path: str) -> str:
    """
    Read the contents of a document from a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The contents of the document.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            document = file.read()
        return document

    except FileNotFoundError:
        logger.exception(f"File not found: {file_path}")
        return ""