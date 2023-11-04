
print("START")
print("Import Libraries")
import json
import textwrap
# Utils
import time
import uuid
from typing import List

import numpy as np
import vertexai
# Vertex AI
from google.cloud import aiplatform
print(f"Vertex AI SDK version: {aiplatform.__version__}")
import os

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the service account JSON file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/user/iconnect/key.json"


# Langchain
import langchain

print(f"LangChain version: {langchain.__version__}")

from langchain.chains import RetrievalQA
from langchain.document_loaders import GCSDirectoryLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

# Import custom Matching Engine packages
from aiutils.matching_engine import MatchingEngine
from aiutils.matching_engine_utils import MatchingEngineUtils
import threading


#custom template formatter
def formatter(result):
    print(f"Query: {result['query']}")
    print("." * 80)
    if "source_documents" in result.keys():
        for idx, ref in enumerate(result["source_documents"]):
            print("-" * 80)
            print(f"REFERENCE #{idx}")
            print("-" * 80)
            if "score" in ref.metadata:
                print(f"Matching Score: {ref.metadata['score']}")
            if "source" in ref.metadata:
                print(f"Document Source: {ref.metadata['source']}")
            if "document_name" in ref.metadata:
                print(f"Document Name: {ref.metadata['document_name']}")
            print("." * 80)
            print(f"Content: \n{wrap(ref.page_content)}")
    print("." * 80)
    print(f"Response: {wrap(result['result'])}")
    print("." * 80)

#custom template formatter
def chatresponse(result):
    print(f"Query: {result['query']}")
    print("." * 80)
    metascore = []
    metasource = []
    metadocname = []
    if "source_documents" in result.keys():
        for idx, ref in enumerate(result["source_documents"]):
            print("-" * 80)
            print(f"REFERENCE #{idx}")
            print("-" * 80)
            if "score" in ref.metadata:
                metascore.append(ref.metadata['score'])
                print(f"Matching Score: {ref.metadata['score']}")
            if "source" in ref.metadata:
                metasource.append(ref.metadata['source'])
                print(f"Document Source: {ref.metadata['source']}")
            if "document_name" in ref.metadata:
                metadocname.append(ref.metadata['document_name'])
                print(f"Document Name: {ref.metadata['document_name']}")
            print("." * 80)
            print(f"Content: \n{wrap(ref.page_content)}")
            # break
    print("." * 80)
    print(f"Response: {wrap(result['result'])}")
    print("." * 80)
    result = {"response": wrap(result['result']), "metascore": metascore,metasource:metasource,metadocname:metadocname}

    return result

def wrap(s):
    return "\n".join(textwrap.wrap(s, width=120, break_long_words=False))




# Utility functions for Embeddings API with rate limiting
def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]


print('Elango: Set the Google cloud project details completed !')


PROJECT_ID = "herfy-dev"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=REGION)


print('Elango: Initialize Vertex AI SDK completed !')

# Text model instance integrated with langChain
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

print('Elango: Text model instance integrated with langChain completed !')
# Embeddings API integrated with langChain
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)


print('Elango: Embeddings API integrated with langChain completed !')

print("STEP 1: Create Matching Engine Index and Endpoint for Retrieval")
ME_REGION = "us-central1"
ME_INDEX_NAME = f"{PROJECT_ID}-me-index"  # @param {type:"string"}
ME_EMBEDDING_DIR = f"{PROJECT_ID}-me-bucket"  # @param {type:"string"}
ME_DIMENSIONS = 768  # when using Vertex PaLM Embedding

mengine = MatchingEngineUtils(PROJECT_ID, ME_REGION, ME_INDEX_NAME)

index = mengine.create_index(
    embedding_gcs_uri=f"gs://{ME_EMBEDDING_DIR}/init_index",
    dimensions=ME_DIMENSIONS,
    index_update_method="streaming",
    index_algorithm="tree-ah",
)
if index:
    print(index.name)
print('Elango: Created index for matching engine completed !')

index_endpoint = mengine.deploy_index()

print("STEP 2: Add Document Embeddings to Matching Engine - Vector Store")
GCS_BUCKET_DOCS = f"{PROJECT_ID}-documents"
folder_prefix = "documents/google-research-pdfs/"
# Ingest PDF files
print('Elango: Deployed index for matching engine completed !')

print(f"Processing documents from {GCS_BUCKET_DOCS}")
loader = GCSDirectoryLoader(
    project_name=PROJECT_ID, bucket=GCS_BUCKET_DOCS, prefix=folder_prefix
)
documents = loader.load()

# Add document name and source to the metadata
for document in documents:
    doc_md = document.metadata
    document_name = doc_md["source"].split("/")[-1]
    # derive doc source from Document loader
    doc_source_prefix = "/".join(GCS_BUCKET_DOCS.split("/")[:3])
    doc_source_suffix = "/".join(doc_md["source"].split("/")[4:-1])
    source = f"{doc_source_prefix}/{doc_source_suffix}"
    document.metadata = {"source": source, "document_name": document_name}

print(f"# of documents loaded (pre-chunking) = {len(documents)}")


# split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)
doc_splits = text_splitter.split_documents(documents)

print('Elango: Splited the documents into chunks !')

# Add chunk number to metadata
for idx, split in enumerate(doc_splits):
    split.metadata["chunk"] = idx

print(f"# of documents = {len(doc_splits)}")

print('Elango: Added chunk number to metadata !')

# ME_INDEX_ID="projects/148158706903/locations/us-central1/indexes/2650372778754048000"
# ME_INDEX_ENDPOINT_ID="projects/148158706903/locations/us-central1/indexEndpoints/391817565627744256"


ME_INDEX_ID, ME_INDEX_ENDPOINT_ID = mengine.get_index_and_endpoint()
print(f"ME_INDEX_ID={ME_INDEX_ID}")
print(f"ME_INDEX_ENDPOINT_ID={ME_INDEX_ENDPOINT_ID}")

print("Elango: initialize vector store completed !")
# initialize vector store
me = MatchingEngine.from_components(    
    project_id=PROJECT_ID,
    region=ME_REGION,
    gcs_bucket_name=f"gs://{ME_EMBEDDING_DIR}".split("/")[2],
    embedding=embeddings,
    index_id=ME_INDEX_ID,
    endpoint_id=ME_INDEX_ENDPOINT_ID,
)



print("Elango: Store docs as embeddings in Matching Engine index !")

# Store docs as embeddings in Matching Engine index
# It may take a while since API is rate limited
texts = [doc.page_content for doc in doc_splits]
metadatas = [
    [
        {"namespace": "source", "allow_list": [doc.metadata["source"]]},
        {"namespace": "document_name", "allow_list": [doc.metadata["document_name"]]},
        {"namespace": "chunk", "allow_list": [str(doc.metadata["chunk"])]},
    ]
    for doc in doc_splits
]

print("Elango: Added Embedding for vector store !")


doc_ids = me.add_texts(texts=texts, metadatas=metadatas)




print("Elango: Validate semantic search with Matching Engine is working !")
# ssearch = me.similarity_search("Budget india summary", k=2)

# print(ssearch)

print("STEP 3: Retrieval based Question/Answering Chain")

print("Elango: Configure Question/Answering Chain with Vector Store using Text")

print("Elango: Create chain to answer questions")
# Create chain to answer questions
NUMBER_OF_RESULTS = 3
SEARCH_DISTANCE_THRESHOLD = 0.6

# Expose index to the retriever
retriever = me.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": NUMBER_OF_RESULTS,
        "search_distance": SEARCH_DISTANCE_THRESHOLD,
    },
)


print("Elango: Customize the default retrieval prompt template")

template = """
=============
{context}
=============

Question: {question}
Helpful Answer:"""

print("Elango: Configure RetrievalQA chain")
print("Use Vertex PaLM Text API for LLM!")

# Uses LLM to synthesize results from the search index.
# Use Vertex PaLM Text API for LLM
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        ),
    },
)

print("Elango: Going to Run QA chain on questions")

def ask(query, qa=qa, k=NUMBER_OF_RESULTS, search_distance=SEARCH_DISTANCE_THRESHOLD):
    print("query: "+query)
    print("query qa: "+str(qa))
    print("query k: "+str(k))
    print("query search_distance: "+str(search_distance))
    qa.retriever.search_kwargs["search_distance"] = search_distance
    qa.retriever.search_kwargs["k"] = k
    result = qa({"query": query})
    return formatter(result)

def askchat(query, qa=qa, k=NUMBER_OF_RESULTS, search_distance=SEARCH_DISTANCE_THRESHOLD):
    print("query: "+query)
    print("query qa: "+str(qa))
    print("query k: "+str(k))
    print("query search_distance: "+str(search_distance))
    qa.retriever.search_kwargs["search_distance"] = search_distance
    qa.retriever.search_kwargs["k"] = k
    result = qa({"query": query})
    return chatresponse(result)    

def chat(query):
    k = 3
    search_distance = 0.6
    print("query: "+query)
    # print("query qa: "+str(qa))
    print("query k: "+str(k))
    print("query search_distance: "+str(search_distance))
    qa.retriever.search_kwargs["search_distance"] = search_distance
    qa.retriever.search_kwargs["k"] = k
    result = qa({"query": query})
    return formatter(result)
    
# result = ask("Budget india summary")
# print(result)
print("END")
"""https://colab.research.google.com/drive/1T6x83-wGzcViH9Efm57ygtMLwgFQJH1d#scrollTo=OJecKul0xgWT """



