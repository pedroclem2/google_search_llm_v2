import requests
from ..configs.service_config import *

def fetch_urls_from_query(search_query, num_results=10):
    """
    Fetch the URLs from Google Custom Search based on the provided query,
    and return a list of URLs for the top search results. This function will
    return the specified number of results if available.
    
    Parameters:
        search_query (str): The query to search for.
        num_results (int): The number of search results to return.
    
    Returns:
        list: A list of URLs from the top search results.
    
    Raises:
        Exception: If no results are found or there is an issue with the request.
    """
    #configs
    google_api_key = google_search_api_key()
    cx = get_search_engine_cx()
    
    #parameters
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q': search_query,
        'key': google_api_key,
        'cx': cx,
        'num': num_results  #multiple results
    }

    
    response = requests.get(url, params=params)
    results = response.json()

   
    if 'items' in results:
        urls = [item['link'] for item in results['items']]
        return urls
    else:
        raise Exception("No results found for the query.")
    
    

if __name__ == "__main__":
    try:
        search_query = "what kind of wine goes well with beef wellington?"
        url = fetch_urls_from_query(search_query)
        print(url)  
    except Exception as e:
        print(str(e))