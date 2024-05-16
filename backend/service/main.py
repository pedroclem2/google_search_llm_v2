import asyncio
from loguru import logger
from service.configs.service_config import *
from service.engines.loader import load_and_transform
from service.engines.image_processor import WineImageAnalyzer

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader

async def main():
    """
    Main function to execute the load, transform, and query pipeline.
    Allows user to input either text or upload an image of a wine label.
    """
    input_type = input("Type 'text' to enter a query or 'image' to upload an image: ").strip().lower()
    analyzer = WineImageAnalyzer()  #instantiate the image analyzer
    
    if input_type == 'text':
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
        

        embeddings = OpenAIEmbeddings(deployment="text-embedding-3-small", chunk_size=500)
        docsearch = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="service/databases/chromadb")
        llm = ChatOpenAI()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=docsearch.as_retriever(), return_source_documents=False)
        response = qa.run(search_query)
        logger.info(response)
    
    elif input_type == 'image':

        folder_path = "service/databases/images"
        image_desc = analyzer.analyze_images_in_folder(folder_path)
        explainer = f"Image descriptions: {image_desc} \n\n If the provided image description is of a wine, give me back further deatails of that wine and with what foods it would pair well with. If the provided image description is of a food, give back which wine(s) would pair best with that food."
        final_answer = analyzer.gpt4o_chat(explainer)
        print(f"{final_answer}")

if __name__ == "__main__":
    asyncio.run(main())
