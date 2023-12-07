# https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/expositional/vector-dbs/milvus/milvus_evals_build_better_rags.ipynb
# https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/expositional/vector-dbs/milvus/milvus_simple.ipynb
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/MilvusIndexDemo.html#milvus-vector-store
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import MilvusVectorStore
from llama_index.llms import OpenAI

from llama_index import (
    VectorStoreIndex,
    # SimpleWebPageReader,
    LLMPredictor,
    ServiceContext
)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

from tenacity import retry, stop_after_attempt, wait_exponential

from trulens_eval import TruLlama, Feedback, Tru, feedback
from trulens_eval.feedback import Groundedness


tru = Tru()

# from llama_index import WikipediaReader

cities = [
    "Los Angeles", "Houston", "Honolulu", "Tucson", "Mexico City", 
    "Cincinatti", "Chicago"
]

wiki_docs = []
for city in cities:
    try:
        doc = WikipediaReader().load_data(pages=[city])
        wiki_docs.extend(doc)
    except Exception as e:
        print(f"Error loading page for city {city}: {e}")


test_prompts = [
    "What's the best national park near Honolulu",
    "What are some famous universities in Tucson?",
    "What bodies of water are near Chicago?",
    "What is the name of Chicago's central business district?",
    "What are the two most famous universities in Los Angeles?",
    "What are some famous festivals in Mexico City?",
    "What are some famous festivals in Los Angeles?",
    "What professional sports teams are located in Los Angeles",
    "How do you classify Houston's climate?",
    "What landmarks should I know about in Cincinatti"
]


vector_store = MilvusVectorStore(index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "L2"
        },
        search_params={"nprobe": 20},
        overwrite=True)
llm = OpenAI(model="gpt-3.5-turbo")
embed_v12 = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
storage_context = StorageContext.from_defaults(vector_store = vector_store)
service_context = ServiceContext.from_defaults(embed_model = embed_v12, llm = llm)
index = VectorStoreIndex.from_documents(wiki_docs,
            service_context=service_context,
            storage_context=storage_context)
query_engine = index.as_query_engine(top_k = 5)

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_query_engine(prompt):
        return query_engine.query(prompt)
for prompt in test_prompts:
    call_query_engine(prompt)


import numpy as np

# Initialize OpenAI-based feedback function collection class:
openai_gpt35 = feedback.OpenAI(model_engine="gpt-3.5-turbo")

# Define groundedness
grounded = Groundedness(groundedness_provider=openai_gpt35)
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness").on(
    TruLlama.select_source_nodes().node.text.collect() # context
).on_output().aggregate(grounded.grounded_statements_aggregator)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai_gpt35.relevance_with_cot_reasons, name = "Answer Relevance").on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai_gpt35.qs_relevance_with_cot_reasons, name = "Context Relevance").on_input().on(
    TruLlama.select_source_nodes().node.text
).aggregate(np.max)


index_params = ["IVF_FLAT","HNSW"]
embed_v12 = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embed_ft3_v12 = HuggingFaceEmbeddings(model_name = "Sprylab/paraphrase-multilingual-MiniLM-L12-v2-fine-tuned-3")
embed_ada = OpenAIEmbeddings(model_name = "text-embedding-ada-002")
embed_models = [embed_v12, embed_ada]
top_ks = [1,3]
chunk_sizes = [200,500]

import itertools
for index_param, embed_model, top_k, chunk_size in itertools.product(
    index_params, embed_models, top_ks, chunk_sizes
    ):
    if embed_model == embed_v12:
        embed_model_name = "v12"
    elif embed_model == embed_ft3_v12:
        embed_model_name = "ft3_v12"
    elif embed_model == embed_ada:
        embed_model_name = "ada"
    vector_store = MilvusVectorStore(index_params={
        "index_type": index_param,
        "metric_type": "L2"
        },
        search_params={"nprobe": 20},
        overwrite=True)
    llm = OpenAI(model="gpt-3.5-turbo")
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    service_context = ServiceContext.from_defaults(embed_model = embed_model, llm = llm, chunk_size=chunk_size)
    index = VectorStoreIndex.from_documents(wiki_docs,
            service_context=service_context,
            storage_context=storage_context)
    query_engine = index.as_query_engine(similarity_top_k = top_k)
    tru_query_engine = TruLlama(query_engine,
                    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance],
                    metadata={
                        'index_param':index_param,
                        'embed_model':embed_model_name,
                        'top_k':top_k,
                        'chunk_size':chunk_size
                        })
    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_tru_query_engine(prompt):
        return tru_query_engine.query(prompt)
    for prompt in test_prompts:
        call_tru_query_engine(prompt)

data = tru.get_records_and_feedback(app_ids=[])[0]
x=0