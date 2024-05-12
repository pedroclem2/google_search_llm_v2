import asyncio
import os
from loguru import logger
from ..engines.url_fetcher import *
from ..configs.service_config import get_openai_api_key

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader

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

async def main():
    """
    Main function to execute the load, transform, and query pipeline.
    """
    search_query = input("Enter your search query: ")
    filenames = await load_and_transform(search_query)

    if not filenames:
        logger.error("No content available after multiple attempts.")
        return

    loader = DirectoryLoader('service/databases/raw_files', glob="*.txt")
    docs = loader.load()
    if not docs:
        raise ValueError("No documents found in the directory.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    os.environ["OPENAI_API_KEY"] = get_openai_api_key()
    embeddings = OpenAIEmbeddings(deployment="text-embedding-3-small", chunk_size=500)
    docsearch = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="service/databases/chromadb")

    llm = ChatOpenAI()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=docsearch.as_retriever(), return_source_documents=False)

    response = qa.run(search_query)
    logger.info(response)

if __name__ == "__main__":
    asyncio.run(main())
