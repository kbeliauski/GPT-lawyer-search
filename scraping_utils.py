import requests
from bs4 import BeautifulSoup

def parse_page(url):
    page = requests.get(url.strip())
    soup = BeautifulSoup(page.content, 'lxml')
    return soup.body.get_text()