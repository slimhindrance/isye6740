import re
from bs4 import BeautifulSoup

def clean_text(text):
    """
    Cleans up a messy text string to make it more readable as a conversation.
    - Removes HTML tags and XML-like elements
    - Converts formatted text into plain readable sentences
    - Removes excessive whitespace and newlines
    - Preserves the conversation structure

    Args:
        text (str): The raw input text containing messy formatting.

    Returns:
        str: Cleaned and readable conversation text.
    """

    # ✅ Step 1: Remove HTML tags and XML-like elements
    text = BeautifulSoup(text, "html.parser").get_text()

    # ✅ Step 2: Replace unwanted elements
    text = re.sub(r"\[.*?\]", "", text)  # Remove anything in square brackets
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces/newlines with a single space
    text = text.replace("<break/>", "\n")  # Convert breaks into actual new lines

    # ✅ Step 3: Handle unwanted artifacts
    text = text.replace("paragraph>", "")  # Remove redundant paragraph indicators
    text = re.sub(r"Figure:\s*", "", text)  # Remove "Figure:" indicators
    text = re.sub(r"^\s+|\s+$", "", text)  # Trim leading and trailing whitespace

    return text