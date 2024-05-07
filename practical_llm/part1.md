# Practical LLM: Two easy methods to boost your RAG metrics

Retrieval-Augmented Generation (RAG) is a framework used to provide additional knowledge to LLMs in order to extend their context knowledge and enable them to return more adequate answers. In particular, companies this enables companies to feed their internal data to a LLM and get personalized responses.
The RAG process can be divided into 3 phases:
- indexing : the indexing of external data usually consists in chunking the text extracted from documents, computing their embedding and storing them in a special type of database known as `VectorStores`.
- retrieval : during retrieval, the embedding of the query is computed and chunks that are the most relevant for it are retrieved in order to serve as context in the generation prompt 
- generation : finally leveraging the context that was retrieved, the LLM will formulate its answer.


## Naive RAG implementation in LLamax-index

### Prerequisites

This code has been run with Python 3.11.7 with the following `requirements.txt` file:
```py
jupyter
llama-index==0.10.34
llama-index-embeddings-huggingface==0.2.0
openai==1.25.1
python-dotenv #reads .env file
trulens-eval==0.28.2 #used to evaluate RAGs
```
and defining those variables in a `.env` file:
```py
OPENAI_API_KEY=<insert your openai api key>
HUGGINGFACE_API_KEY=<insert your huggingface api key>
``` 

### Get Data

We will use a Paul Graham's essay for illustration purposes here is how to download it:
```py
!mkdir -p '../data'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O '../data/paul_graham_essay.txt'
```

### Naive RAG Implementation

Now that we have some text file to work with let us proceed with building a naive RAG. 

For this experiment, we will use OpenAI for the LLM and BGE as embedding model. `ServiceContext` has been deprecated since the version v.0.10.0 of llama-index and replaced by `Settings`.
The `Settings` object enables to set global parameters and those are lazily instantiated. This means that only required elements will be loaded into memory. [See doc.](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/)

```py
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```

From there, it only takes 4 rows to get a Naive RAG running.

```py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


documents = SimpleDirectoryReader(
    input_files=["../data/paul_graham_essay.txt"]
).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What did Paul Graham study?")
str(response)
```

## Limitations and Causes

Naive RAG are a fast way to get started interacting with your documents but the questions they are able to answer are only as good as the context it can retrieve. That is why the `chunk_size` parameter is usually subject to tuning as it impacts drastically the performances. 
However, other approaches revealed themselves to retrieve the most relevant chunks. Namely the `Sentence Window Retrieval`.


## Small-to-big retrieval

## 

## Summary



