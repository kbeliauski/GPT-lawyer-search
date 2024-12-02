from llm_utils import llm, get_embedding
from scraping_utils import parse_page
import csv
import json
import os
from tiktoken import encoding_for_model
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm

CACHE_FILE = 'lawyer_profiles.json'
QUERY_CACHE_FILE = 'query_results.json'
TOKEN_LIMIT = 4000  # Adjust based on the model (e.g., 4096 for GPT-3.5-turbo)
EMBEDDING_MODEL = "text-embedding-ada-002"

# Token cache for faster repeated calculations
token_cache = {}


def load_lawyer_links(file_path: str) -> list:
    """
    Load lawyer profile links from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        list: List of URLs to lawyer profiles.
    """
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]


def scrape_profile(link):
    """
    Scrape a single lawyer profile.
    Args:
        link (str): URL to scrape.
    Returns:
        tuple: The URL and its scraped text content.
    """
    try:
        return link, parse_page(link)
    except Exception as e:
        print(f"Error scraping {link}: {e}")
        return link, None


def scrape_and_cache_profiles(lawyer_links: list) -> dict:
    """
    Scrape and cache lawyer profiles into a JSON file.
    Args:
        lawyer_links (list): List of URLs to lawyer profiles.
    Returns:
        dict: Cached profiles with URLs as keys and profile text as values.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as cache_file:
            print("Loading cached profiles...")
            return json.load(cache_file)

    print("Scraping lawyer profiles... This may take some time.")
    profiles = {}
    # Adjust max_workers based on system resources
    with ThreadPoolExecutor(max_workers=30) as executor:
        results = executor.map(scrape_profile, lawyer_links)

    for url, text in results:
        if text:
            profiles[url] = text

    with open(CACHE_FILE, 'w') as cache_file:
        json.dump(profiles, cache_file)
    print("Profiles cached successfully.")

    return profiles


def estimate_tokens_cached(text: str, model: str = "gpt-4") -> int:
    """
    Estimate the number of tokens in a given text using OpenAI's tokenizer.
    Args:
        text (str): The text to tokenize.
        model (str): The model to use for token estimation.
    Returns:
        int: The estimated number of tokens.
    """
    if text in token_cache:
        return token_cache[text]
    encoder = encoding_for_model(model)
    tokens = len(encoder.encode(text))
    token_cache[text] = tokens
    return tokens


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    Args:
        vec1 (numpy.ndarray): First vector.
        vec2 (numpy.ndarray): Second vector.
    Returns:
        float: Cosine similarity score.
    """
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def filter_profiles_with_embeddings(profiles: dict, query: str) -> dict:
    """
    Filter profiles using cosine similarity based on embeddings.
    Args:
        profiles (dict): Cached profiles with URLs as keys and profile text as values.
        query (str): The query to evaluate against.
    Returns:
        dict: Filtered profiles that meet the similarity threshold.
    """
    query_embedding = get_embedding([query], size=EMBEDDING_MODEL)[0]
    filtered_profiles = {}
    for url, text in profiles.items():
        profile_embedding = get_embedding([text], size=EMBEDDING_MODEL)[0]
        similarity = cosine_similarity(query_embedding, profile_embedding)
        if similarity > 0.8:  # Adjust threshold as needed
            filtered_profiles[url] = text
    return filtered_profiles


def create_batch(profile_items, i, token_budget, model):
    """
    Create a batch of profiles within the token limit.
    Args:
        profile_items (list): List of profiles (URL, text) tuples.
        i (int): Starting index.
        token_budget (int): Remaining token budget.
        model (str): Model to estimate token usage.
    Returns:
        tuple: Batch of profiles and updated index.
    """
    current_tokens = 0
    batch = []

    while i < len(profile_items) and current_tokens < token_budget:
        url, text = profile_items[i]
        profile_tokens = estimate_tokens_cached(f"<{url}>: {text}", model)
        if current_tokens + profile_tokens <= token_budget:
            batch.append((url, text))
            current_tokens += profile_tokens
            i += 1
        else:
            break

    return batch, i


def retry_batch(system_prompt, batch_prompt, retries=3):
    """
    Retry a batch request in case of failure.
    Args:
        system_prompt (str): The system-level prompt.
        batch_prompt (str): The batch-specific prompt.
        retries (int): Number of retries.
    Returns:
        str: The response or None if failed.
    """
    for attempt in range(retries):
        try:
            return llm(system_prompt=system_prompt, user_prompt=batch_prompt)
        except Exception as e:
            print(f"Retry {attempt + 1} failed: {e}")
    print("Batch failed after retries.")
    return None


def batch_passes_criteria(profiles: dict, query: str, model: str = "gpt-4"):
    """
    Evaluate profiles in batches against a given query.
    Args:
        profiles (dict): Cached profiles with URLs as keys and profile text as values.
        query (str): The query to evaluate against.
        model (str): Model to use for token estimation.
    Returns:
        list: A list of URLs for profiles that pass the criterion.
    """
    matching_lawyers = []
    system_prompt = """
    You are evaluating multiple lawyer profiles against the following query. For each profile, respond in this format:

    <lawyer_url>: Pass/Fail

    Only include "Pass" or "Fail" for each URL, based on whether the profile matches the query.
    """.strip()

    token_budget = TOKEN_LIMIT - \
        estimate_tokens_cached(system_prompt, model) - \
        estimate_tokens_cached(query, model)
    profile_items = list(profiles.items())
    i = 0

    with tqdm(total=len(profile_items)) as pbar:  # Add progress tracking
        while i < len(profile_items):
            batch, i = create_batch(profile_items, i, token_budget, model)
            batch_prompt = f"Query: {query}\n\nProfiles:\n" + \
                "\n".join(f"<{url}>: {text}" for url, text in batch)

            response = retry_batch(system_prompt, batch_prompt)
            if response:
                matching_lawyers.extend(result.split(":")[0].strip(
                    "<>") for result in response.splitlines() if "Pass" in result)
            pbar.update(len(batch))

    return matching_lawyers


def cache_query_results(query: str, results: list):
    """
    Cache query results to avoid repeated processing.
    Args:
        query (str): The search query.
        results (list): List of matching profiles.
    """
    if os.path.exists(QUERY_CACHE_FILE):
        with open(QUERY_CACHE_FILE, 'r') as cache_file:
            query_cache = json.load(cache_file)
    else:
        query_cache = {}

    query_cache[query] = results
    with open(QUERY_CACHE_FILE, 'w') as cache_file:
        json.dump(query_cache, cache_file)


def get_cached_results(query: str):
    """
    Retrieve cached results for a query.
    Args:
        query (str): The search query.
    Returns:
        list or None: Cached results or None if not found.
    """
    if os.path.exists(QUERY_CACHE_FILE):
        with open(QUERY_CACHE_FILE, 'r') as cache_file:
            query_cache = json.load(cache_file)
            return query_cache.get(query, None)
    return None


def main(query: str, profiles: dict) -> list:
    """
    Main function to process the query and find matching lawyers.
    Args:
        query (str): The search query.
        profiles (dict): Cached profiles with URLs as keys and profile text as values.
    Returns:
        list: List of matching profiles.
    """
    cached_results = get_cached_results(query)
    if cached_results:
        print(f"Found cached results for query '{query}'.")
        return cached_results

    filtered_profiles = filter_profiles_with_embeddings(profiles, query)
    results = batch_passes_criteria(filtered_profiles, query)
    cache_query_results(query, results)
    return results


if __name__ == '__main__':
    lawyer_links = load_lawyer_links('lawyers.csv')
    profiles = scrape_and_cache_profiles(lawyer_links)

    while True:
        user_query = input(
            "\nEnter your search term (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        print("Evaluating lawyers in batches... This may take some time.")
        results = main(user_query, profiles)
        print(f"\nFound {len(results)} matching lawyers:")
        for result in results:
            print(result)
