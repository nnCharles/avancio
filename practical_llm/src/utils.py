from dotenv import load_dotenv, find_dotenv
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex
)
from llama_index.core.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank
)
from llama_index.core.node_parser import (
    get_leaf_nodes,
    HierarchicalNodeParser,
    SentenceWindowNodeParser
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
import numpy as np
import openai
import os
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)
from trulens_eval.feedback import Groundedness

def setup():
    _ = load_dotenv(find_dotenv())
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    except:
        print("OPENAI_API_KEY not found,")

def get_feedback_func():
    #Init the provider
    oai = OpenAI()
    qa_relevance = (
        Feedback(
            oai.relevance_with_cot_reasons,
            name="Answer Relevance"
        )
        .on_input_output()
    )
    qs_relevance = (
        Feedback(
            oai.relevance_with_cot_reasons,
            name = "Context Relevance"
        )
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )
    grounded = Groundedness(groundedness_provider=oai)
    groundedness = (
        Feedback(
            grounded.groundedness_measure_with_cot_reasons,
            name="Groundedness"
        )
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )
    feedbacks = [qa_relevance, qs_relevance, groundedness]
    return feedbacks

def evaluate_engine(
    query_engine,
    feedbacks,
    app_id,
    eval_questions,
    verbose=False
):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    for question in eval_questions:
        with tru_recorder:
            response = query_engine.query(question)
            if verbose:
                print(question)
                print(str(response))
    return tru_recorder

def build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=1,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            document,
            node_parser=node_parser,
            llm=llm,
            embed_model=embed_model
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
        )
    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, 
        model="BAAI/bge-reranker-base"
    )
    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, 
        node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine

def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=[2048, 512, 128]
):
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, 
            storage_context=storage_context, 
            llm=llm,
            embed_model=embed_model,
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            llm=llm,
            embed_model=embed_model
        )
    return automerging_index

def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=10,
    rerank_top_n=3,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, 
        automerging_index.storage_context,
        verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n,
        model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[rerank]
    )
    return auto_merging_engine
