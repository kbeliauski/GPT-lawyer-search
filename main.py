from llm_utils import llm, get_embedding
from scraping_utils import parse_page

def passes_criterion(lawyer_url: str, query: str) -> bool:
    """
    Evaluate if a lawyer passes a given criterion based on their profile.

    Args:
        lawyer_url (str): URL of the lawyer's profile
        query (str): Criterion to evaluate against

    Returns:
        bool: True if lawyer passes the criterion, False otherwise
    """
    text = parse_page(lawyer_url)
    
    system_prompt = """
    You are evaluating a lawyer whether they pass a given criterion.
    
    Respond in the following format:
    <thinking>...</thinking>, within which you include your detailed thought process.
    <answer>...</answer>, within which you include your final answer. "Pass" or "Fail".
    """.strip()
    
    user_prompt = f"""
    Here is the query: {query}
    Here is the lawyer's profile: {text}
    """.strip()
    
    response = llm(system_prompt=system_prompt, user_prompt=user_prompt)
    return response.split('<answer>')[1].split('</answer>')[0].strip() == 'Pass'

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
