from llm_utils import llm, get_embedding
from scraping_utils import parse_page

def main(query: str) -> list:
    """
    Takes in a string as a query and returns the list of lawyers.

    Args:
        query (str): The search query.

    Returns:
        list: A list of lawyers matching the query.
    """
    # TODO: Implement the search functionality
    return []

if __name__ == '__main__':
    user_query = input('Enter your search term: ')
    print(main(user_query))
