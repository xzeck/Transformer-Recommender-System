# THIS IS AN EXTRACTED PY FILE FROM A JUPYTER NOTEBOOK - SOME CODE ARE REPEATED

import pandas as pd
import csv

# Reading and printing the csv file as a data frame

df = pd.read_csv('flipkart_com-ecommerce_sample.csv')

df.head()

# Removing unwanted columns which we are not processing
df.drop(columns=['crawl_timestamp', 'product_url', 'product_category_tree', 'pid', 'image', 'is_FK_Advantage_product', 'product_rating', 'overall_rating', 'product_specifications'], inplace=True)
df.head()

# Removing any rows that are needed but doesn't have any data
df = df[df['description'].notna() & (df['description'].astype(str).str.len() > 0)]
df = df[df['product_name'].notna() & (df['product_name'].astype(str).str.len() > 0)]
df

# Removing stop words from the description
# This is so that we can save some space and remove unnecessary words from the description

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    word_tokens = word_tokenize(text)

    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]

    return ' '.join(filtered_text)

df['description'] = df['description'].apply(remove_stopwords)

print(df['description'][6])
print(df['product_name'][6])

# Generating LLM context column which is all the things required to give to the LLM as a context

df['llm_context'] = (
    "Product Name: " + df['product_name'] +
    ",\nBrand: " + df['brand'] +
    ",\nDescription: " + df['description'] +
    ",\nRetail Price: " + df['retail_price'].astype(str)
)

print(df['llm_context'][6])

# Dropping description column as its in the context

df.drop('description', axis=1, inplace=True)
df.drop('uniq_id', axis=1, inplace=True)


df.head()

from langchain_community.document_loaders import DataFrameLoader

# Loading the data frame and sepcifying the column which contains page context
loader = DataFrameLoader(df, page_content_column='llm_context')
dataset_docs = loader.load()

# Getting a subset of the documnets
dataset_docs_subset = dataset_docs[:500]

print("PAGE CONTENT:")
print(dataset_docs_subset[6].page_content)
print("\n\n----------------------------\n")
print("METADATA:")
print(dataset_docs_subset[6].metadata)

EMB_MODEL = "thenlper/gte-small" # Model used for embedding
CHUNK_SIZE = 512 # CHUNK_SIZE is set to 512 which gives a good amount of context and is within Llama3's context window

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

# Initializing the embedding tokenizer
embedding_tokenzier = AutoTokenizer.from_pretrained(EMB_MODEL)

# Splitting the document by chunk_size, KB = Knowledge Base
def split_documents(chunk_size, KB, tokenizer=embedding_tokenzier):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embedding_tokenzier, # tokenizer with embedding model
        chunk_size=CHUNK_SIZE, # Chunk size = 512
        chunk_overlap=int(chunk_size / 10), # Sliding window approach for having more context
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS
    )

    # Iterating through all the documents and splitting the document into chunks and adding it to docs_processed
    docs_processed = []
    for doc in KB:
      docs_processed += text_splitter.split_documents([doc])

    # Building a new list of unique documents
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

docs_processed_tok = split_documents(
    CHUNK_SIZE,
    dataset_docs,
    EMB_MODEL
)

docs_processed_tok[0].page_content

docs_processed_tok[0].metadata

from langchain_community.embeddings import HuggingFaceEmbeddings

# Initializing the embedding model
# We are using the huggingface embedding model which is specifically built for this task
embedding_model = HuggingFaceEmbeddings(
    model_name = EMB_MODEL, # Use the defined embedding model
    multi_process = True,
    model_kwargs={"device": "cuda"}, # using GPU so that it can perform faster
    encode_kwargs={"normalize_embeddings": True}, # Normalize the embedding so it has a magnitude of 1
)

from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.runnables import ConfigurableField
from langchain_community.vectorstores.utils import DistanceStrategy

num_docs = 5
# We are using BM25 and FAISS to rank the documents based on how relevant are they to the search query


# Initializing BM25 retriever
bm25_retriever = BM25Retriever.from_documents(
    docs_processed_tok # tokens for the documents
    ).configurable_fields(
    k=ConfigurableField(
        id="search_kwargs_bm25",
        name="k",
        description="The search kwargs to use",
    )
)

# Initializing FAISS retriever
faiss_vectorstore = FAISS.from_documents(
    docs_processed_tok, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

faiss_retriever = faiss_vectorstore.as_retriever(
    search_kwargs={"k": num_docs}
    ).configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs_faiss",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

# EnsembleRetriever is used to combine multiple retrievers together
# It combines the results from bm25 and faiss, each contributes equally
# which is defined by the weights=[0.5, 0.5]
vector_database = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

user_query = """
How much 'ml' does does Sicons Dog Shampoo contain?
"""

config = {"configurable": {"search_kwargs_faiss": {"k": 5}, "search_kwargs_bm25": 5}}
retrieved_docs = vector_database.invoke(user_query, config=config)

print("Top document content match:")
print(retrieved_docs[0].page_content)
print("\n\nTop document metadata match")
print(retrieved_docs[0].metadata)

for docs in retrieved_docs:
  print(docs.metadata)

user_query = """
Recommend me a beard wash that has Aloevera
"""

config = {"configurable": {"search_kwargs_faiss": {"k": 5}, "search_kwargs_bm25": 5}}
retrieved_docs = vector_database.invoke(user_query, config=config)

print("Top document content match:")
print(retrieved_docs[0].page_content)
print("\n\nTop document metadata match")
print(retrieved_docs[0].metadata)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# we are using llama3-8b version, which is large so we will use the 4 bit quantized
# version of it
# in the bnb config we set load_in_4bit as true so it can load the quantized version
# The quanitzation type is nf4 and we define the compute type to be bfloat16 which
# will only use 16 bit floats for precision
# This is done for effeciency in terms of GPU usage


# using llama3-8billion 4bit quantized version
model = "unsloth/llama-3-8b-bnb-4bit"

# BitsandBytes config
bnbConfig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# bnbConfig = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0,
#     llm_int8_has_fp16_weight=False,
# )


# creating a tokenizer for the llama modle
tokenizer = AutoTokenizer.from_pretrained(model)
# tokenizer.pad_token = tokenizer.eos_token

# loading the llama model
# quantization config is the bnb config above
# device_map = auto is so that it can be mapped to all available gpus
# trust_remove_code because sometimes it doesn't trust code downloaded from remote rpositories
# offload_folder so that it can offload any parameters to the folder which can help with memory optimization
model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnbConfig,
    device_map="auto",
    # max_memory=max_memory,
    trust_remote_code=True,
    offload_folder="offload"
)

# padding so that we can ensure compatability during inference step
model.config.pad_token_id = tokenizer.pad_token_id


from IPython.display import display, Markdown

from langchain.prompts.prompt import PromptTemplate

# Creating a prompt template
prompt_template = """
Answer the following QUERY using the provided INFORMATION and CONTEXT. The product recommendation must include the exact product name and price as given in the 'Product Name' and 'Price' in INFORMATION.
Explain why the query fits the recommendation and do not generate or recommend any book title or author that is not explicitly mentioned in INFORMATION.

INFORMATION:
(use this exact product name and price for your answer):
Product Name: {metadata[product_name]}
Price: {metadata[retail_price]}


CONTEXT: {page_content}

QUERY:
{query}

ANSWER:
"""

RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["metadata", "page_content", "query"],
    template=prompt_template,
)

def answer_with_rag(
    query,
    llm,
    knowledge_index: FAISS = vector_database,
    num_retrieved_docs: int = 10,
    num_docs_final: int = 3,
):
    # Gather documents with retriever
    config = {"configurable": {"search_kwargs_faiss": {"k": num_retrieved_docs}, "search_kwargs_bm25": num_retrieved_docs}}
    relevant_docs = knowledge_index.invoke(query, config=config)

    if relevant_docs:
        relevant_docs = relevant_docs[:num_docs_final]
        page_content = relevant_docs[0].page_content
        metadata = relevant_docs[0].metadata
    else:
        page_content = "No relevant context found."
        metadata = {"Product Name": "No product available", "price": "No proice available"}
    print("Documents retrieved")

    # Build the final prompt
    final_prompt = RAG_PROMPT_TEMPLATE.format(
        metadata=metadata,
        page_content=page_content,
        query=query
    )
    print("\nGenerated prompt:")
    print(final_prompt)

    # Redact an answer
    inputs = tokenizer(final_prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, num_return_sequences=1, max_new_tokens=1000)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer generated")

    return answer, relevant_docs

def display_answer(generated_answer, relevant_docs):
    split = generated_answer.split("ANSWER:")
    answer = split[1].strip() if len(split) > 1 else "No answer has been generated."

    display(Markdown("### llama3 Answer"))
    display(Markdown(answer))

    display(Markdown("### Source documents"))
    for i, doc in enumerate(relevant_docs):
        display(Markdown(f"**Document {i + 1}**"))
        display(Markdown(f"**Page Content:** {doc.page_content}"))
        display(Markdown(f"**Metadata:** {doc.metadata}"))

query = """
Can you tell me the price for Sicons Dog Shampoo?
"""

answer, relevant_docs = answer_with_rag(query, model)
display_answer(answer, relevant_docs)

query = """
Can you recommend a men's shirt?
"""

answer, relevant_docs = answer_with_rag(query, model)
display_answer(answer, relevant_docs)

query = """
How much does men's footwear cost?
"""

answer, relevant_docs = answer_with_rag(query, model)
display_answer(answer, relevant_docs)