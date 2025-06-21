# from src.indexer import MedicalDataIndexer

# # Initialize indexer
# indexer = MedicalDataIndexer(index_name="test_indexing")

# # Index your CSV file
# indexer.index_from_csv("data/medical_text_data.csv")  # Adjust path if needed

# # Verify indexing
# doc_count = indexer.document_store.get_document_count()
# print(f"✅ Indexed documents: {doc_count}")

# from src.retriever_reader import QASearchPipeline

# pipeline = QASearchPipeline(index_name="test_indexing")
# query = "What are the side effects of this treatment?"
# query_type = "question"

# result = pipeline.run_pipeline(query, query_type)
# print(result)

#--------------------------------------------------------------------------------------------------------

# from src.ranker import RankerModule

# # Dummy data (replace with actual Haystack Document objects)
# from haystack.schema import Document
# docs = [
#     Document(content="Aspirin can cause stomach upset."),
#     Document(content="Aspirin may reduce blood clotting."),
#     Document(content="Ibuprofen is similar to aspirin.")
# ]

# query = "What are side effects of aspirin?"

# ranker = RankerModule()
# top_docs = ranker.rank_documents(query, docs, top_k=1)

# for doc in top_docs:
#     print(doc.content)

#------------------------------------------------------------------------------

# from src.ranker import AnswerRanker

# ranker = AnswerRanker()
# best = ranker.rank_answers("What are the side effects of aspirin?", reader_results["answers"])
# print("Best Answer:", best)

#-----------------------------------------------------------------------------------

# from src.retriever_reader import QASearchPipeline

# pipeline = QASearchPipeline(index_name="test_indexing")

# query = "What are the side effects of aspirin?"
# query_type = "question"

# # Run the full pipeline
# results = pipeline.run_pipeline(query, query_type)

# # Use the ranker to get the best answer
# best_answer = pipeline.ranker.rank_answers(query, results["answers"])

# print("\n🔍 Query:", query)
# print("🏆 Best Answer:", best_answer.answer)
# print("📊 Score:", best_answer.score)

#-------------------------------------------------------------------------------------

# from src.retriever_reader import QASearchPipeline

# query = "What are the side effects of aspirin?"
# query_type = "question"

# pipeline = QASearchPipeline(index_name="test_indexing")

# # Run pipeline
# results = pipeline.run_pipeline(query, query_type)

# answers = results["answers"]

# print("\n📋 Top Reader Answers:")
# for i, ans in enumerate(answers):
#     print(f"{i+1}. Answer: {ans.answer}")
#     print(f"   Score: {ans.score:.4f}")
#     print(f"   Context: {ans.context[:200]}...\n")

# # Rank using SentenceTransformer-based ranker
# best_answer = pipeline.ranker.rank_answers(query, answers)

# print("🏆 Best Answer by Ranker:")
# print("Answer:", best_answer.answer)
# print("Score:", best_answer.score)

# -----------------------------------------------------------------------------------------

from fastapi import FastAPI
from pydantic import BaseModel
from src.retriever_reader import QASearchPipeline
from src.query_classifier import QueryClassifier

app = FastAPI()
pipeline = QASearchPipeline(index_name="test_indexing")
classifier = QueryClassifier()

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    # Step 1: Classify the query
    query_type = classifier.classify(request.query)

    # Step 2: Run the pipeline
    results = pipeline.run_pipeline(request.query, query_type)

    # Step 3: Return ranked result
    if query_type in ["question", "statement"]:
        best = pipeline.ranker.rank_answers(request.query, results["answers"])
        return {
            "query": request.query,
            "query_type": query_type,
            "best_answer": best.answer,
            "score": best.score,
        }

    # Keyword case: return top document content
    top_doc = results["documents"][0] if results["documents"] else None
    return {
        "query": request.query,
        "query_type": query_type,
        "best_document": top_doc.content if top_doc else None
    }

