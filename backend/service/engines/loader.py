import os
from loguru import logger
from .url_fetcher import *
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer


logger.add("debug.log", rotation="1 week")

async def load_and_transform(search_query):
    """
    Loads and transforms HTML documents fetched from URLs derived from a search query.

    Args:
        search_query (str): The search query to fetch URLs.

    Returns:
        list of str: A list of filenames where the transformed text content is saved. None if no content is saved.
    """
    raw_files_dir = 'service/databases/raw_files'
    os.makedirs(raw_files_dir, exist_ok=True)  

    urls = fetch_urls_from_query(search_query, 10)
    content_files = []

    for url in urls:
        logger.debug(f"Fetched URL: {url}")
        try:
            loader = AsyncHtmlLoader([url])
            docs = loader.load() 
            if not docs:
                logger.warning(f"No documents loaded for URL: {url}")
                continue  #skip to next URL
            html2text = Html2TextTransformer()
            doc_transformed = html2text.transform_documents(docs)
            if doc_transformed and doc_transformed[0].page_content:
                filename = os.path.join(raw_files_dir, f"output_{urls.index(url)}.txt")
                with open(filename, 'w') as file:
                    file.write(doc_transformed[0].page_content)
                logger.info(f"Saved transformed content to {filename}")
                content_files.append(filename)
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")

    return content_files if content_files else None


