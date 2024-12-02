import requests
from bs4 import BeautifulSoup
import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]


def get_random_user_agent():
    return random.choice(USER_AGENTS)


def parse_page(url: str) -> str:
    headers = {"User-Agent": get_random_user_agent()}
    try:
        response = requests.get(url.strip(), headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        return soup.body.get_text()
    except requests.RequestException as e:
        raise requests.RequestException(f"Error fetching the page: {e}")
