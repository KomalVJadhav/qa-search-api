# from fastapi import FastAPI
# from pydantic import BaseModel
# from src.query_classifier import classify_query
# from src.retriever_reader import QASearchPipeline

# app = FastAPI()
# pipeline = QASearchPipeline()

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/search")
# def search(request: QueryRequest):
#     query_type = classify_query(request.query)
#     result = pipeline.run_pipeline(request.query, query_type)
#     return {"query_type": query_type, "results": result}
# ------------------------------------------------------------------------------
# from fastapi import FastAPI
# from pydantic import BaseModel
# from src.retriever_reader import QASearchPipeline

# app = FastAPI()
# pipeline = QASearchPipeline(index_name="test_indexing")

# class QueryRequest(BaseModel):
#     query: str
#     # query_type: str

# @app.post("/search")
# def search(request: QueryRequest):
#     results = pipeline.run_pipeline(request.query, request.query_type)

#     if request.query_type in ["statement", "question"]:
#         best_answer = pipeline.ranker.rank_answers(request.query, results["answers"])
#         return {
#             "query": request.query,
#             "best_answer": best_answer.answer,
#             "score": best_answer.score,
#             "all_answers": [a.answer for a in results["answers"]]
#         }

#     return results
# ------------------------------------------------------------------------------

# from fastapi import FastAPI
# from pydantic import BaseModel
# from src.retriever_reader import QASearchPipeline
# from src.query_classifier import QueryClassifier

# app = FastAPI()
# pipeline = QASearchPipeline(index_name="test_indexing")
# classifier = QueryClassifier()

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/search")
# def search(request: QueryRequest):
#     # Step 1: Classify the query
#     query_type = classifier.classify(request.query)

#     # Step 2: Run the pipeline
#     results = pipeline.run_pipeline(request.query, query_type)

#     # Step 3: Return ranked result
#     if query_type in ["question", "statement"]:
#         best = pipeline.ranker.rank_answers(request.query, results["answers"])
#         return {
#             "query": request.query,
#             "query_type": query_type,
#             "best_answer": best.answer,
#             "score": best.score,
#         }

#     # Keyword case: return top document content
#     top_doc = results["documents"][0] if results["documents"] else None
#     return {
#         "query": request.query,
#         "query_type": query_type,
#         "best_document": top_doc.content if top_doc else None
#     }
#---------------------------------------------------------------------------
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
    query = request.query
    query_type = classifier.classify(query)  # 👈 classifier decides the type

    results = pipeline.run_pipeline(query, query_type)

    if query_type in ["question", "statement"]:
        best = pipeline.ranker.rank_answers(query, results["answers"])
        return {
            "query": query,
            "query_type": query_type,
            "best_answer": best.answer,
            "score": best.score,
        }

    # Keyword case
    top_doc = results["documents"][0] if results["documents"] else None
    return {
        "query": query,
        "query_type": query_type,
        "best_document": top_doc.content if top_doc else None
    }

@app.get("/")
def root():
    return {"message": "QA Search API is running"}
