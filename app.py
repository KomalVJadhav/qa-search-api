from src.indexer import MedicalDataIndexer

indexer = MedicalDataIndexer(index_name="test_indexing")
indexer.index_from_csv("data/medical_text_data.csv")


# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run("src.search_api:app", host="127.0.0.1", port=8000, reload=True)