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

#interaction and output formatting
import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout

async def main():
    """
    Main function to execute the load, transform, and query pipeline.
    Enhanced with inquirer for input and rich for formatted output.
    """
    console = Console()
    layout = Layout()

    questions = [
        inquirer.List('input_type',
                      message="Choose your input type:",
                      choices=['text', 'image'],
                      carousel=True)
    ]
    answers = inquirer.prompt(questions)
    input_type = answers['input_type']

    analyzer = WineImageAnalyzer()  #instantiate the image analyzer

    if input_type == 'text':
        questions = [
            inquirer.Text('search_query', message="Enter your search query:")
        ]
        answers = inquirer.prompt(questions)
        search_query = answers['search_query']

        filenames = await load_and_transform(search_query)
        # logger.debug("Filenames obtained: {}", filenames)

        if not filenames:
            # logger.error("No content available after multiple attempts.")
            return

        loader = DirectoryLoader('service/databases/raw_files', glob="*.txt")
        docs = loader.load()
        if not docs:
            #logger.error("No documents found in the directory.")
            raise ValueError("No documents found in the directory.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(deployment="text-embedding-3-small", chunk_size=500)
        docsearch = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="service/databases/chromadb")
        llm = ChatOpenAI()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=docsearch.as_retriever(), return_source_documents=False)
        response = qa.run(search_query)
        #logger.info("Query response: {}", response)

        #dynamically adjust panel height based on response length
        response_lines = response.count('\n') + 1  #count lines and add one for padding
        min_height = 10
        panel_height = max(response_lines, min_height)

        #display using rich formatting
        layout.split_row(
            Layout(name="query"),
            Layout(name="response")
        )
        layout["query"].update(Panel(f"User query: {search_query}", style="bold cyan", height=min_height))
        layout["response"].update(Panel(f"{response}", title="Query Response", border_style="bold green", height=panel_height))
        console.print(layout)

    elif input_type == 'image':
        folder_path = "service/databases/images"
        image_desc = analyzer.analyze_images_in_folder(folder_path)
        #logger.debug("Image description: {}", image_desc)

        explainer = f"Image descriptions: {image_desc} \n\n If the provided image description is of a wine, give me back further details of that wine and with what foods it would pair well with. If the provided image description is of a food, give back which wine(s) would pair best with that food."
        final_answer = analyzer.gpt4o_chat(explainer)
        # logger.info("Final pairing suggestion: {}", final_answer)

        answer_lines = final_answer.count('\n') + 1
        min_height = 10
        panel_height = max(answer_lines, min_height)

        layout.split_row(
            Layout(name="processed"),
            Layout(name="suggestion")
        )
        layout["processed"].update(Panel(f"Image processed: {folder_path}", style="bold cyan", height=min_height))
        layout["suggestion"].update(Panel(f"{final_answer}", title="Wine/Food Pairing Suggestion", border_style="bold green", height=panel_height))
        console.print(layout)

if __name__ == "__main__":
    asyncio.run(main())
