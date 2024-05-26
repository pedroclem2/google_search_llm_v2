
# Google Search Chat
## Introduction
This project is designed to fetch, load, and transform web content based on user queries, and subsequently process these queries using a language model. It utilizes the Google Custom Search API to retrieve URLs (10) and leverages the LangChain library for text processing and query answering.
***Update:*** Images can now be ingested for search by using the GPT4o model.


### Configuring Environment Variables
Create a `.env` file in the project root directory and populate it with the necessary API keys:
```plaintext
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
SEARCH_ENGINE_CX=your_custom_search_engine_id_here
```

## Usage

CD into backend dir. Then run the application from the terminal:
```bash
python -m service.main
```
Follow the prompts to enter your search query. The system will process the query and return the results after fetching and transforming the content.


