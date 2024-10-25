import requests
from bs4 import BeautifulSoup


def parse_page(url: str) -> str:
    """
    Parse the content of a web page and extract its text.

    Args:
        url (str): The URL of the web page to parse.

    Returns:
        str: The extracted text content of the web page.

    Raises:
        requests.RequestException: If there's an error fetching the page.
    """
    try:
        response = requests.get(url.strip())
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        return soup.body.get_text()
    except requests.RequestException as e:
        raise requests.RequestException(f"Error fetching the page: {e}")