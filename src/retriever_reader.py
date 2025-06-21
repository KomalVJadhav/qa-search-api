from haystack.document_stores import OpenSearchDocumentStore
from haystack.nodes import FARMReader, BM25Retriever
from haystack.pipelines import DocumentSearchPipeline, ExtractiveQAPipeline
from haystack.schema import Document
from src.ranker import AnswerRanker  # ✅ Correct import

class QASearchPipeline:
    def __init__(self, index_name="test_indexing"):
        self.document_store = OpenSearchDocumentStore(
            host="localhost",
            port=9200,
            index=index_name,
            username="",
            password="",
            scheme="http",
            verify_certs=False,
            create_index=False
        )

        self.bm25_retriever = BM25Retriever(document_store=self.document_store)
        self.reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

        self.qa_pipeline = ExtractiveQAPipeline(reader=self.reader, retriever=self.bm25_retriever)
        self.keyword_pipeline = DocumentSearchPipeline(self.bm25_retriever)

        self.ranker = AnswerRanker()  # ✅ Correct class usage


    def run_pipeline(self, query, query_type):
        if query_type == "keyword":
            result = self.keyword_pipeline.run(query=query, params={"Retriever": {"top_k": 10}})
        elif query_type in ["statement", "question"]:
            result = self.qa_pipeline.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}})
            
            # Rank the reader's answers
            ranker = AnswerRanker()
            best_answer = ranker.rank_answers(query, result["answers"])
            result["best_answer"] = {
                "answer": best_answer.answer if best_answer else None,
                "score": best_answer.score if best_answer else None
            }
        else:
            result = {"query": query, "results": []}
        return result
