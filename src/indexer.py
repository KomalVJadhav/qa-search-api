from haystack.document_stores import OpenSearchDocumentStore
from haystack.schema import Document
import pandas as pd

class MedicalDataIndexer:
    def __init__(self, index_name="test_indexing"):
        self.document_store = OpenSearchDocumentStore(
            host="localhost",
            port=9200,
            index=index_name,
            username="",
            password="",
            scheme="http",
            verify_certs=False,
            create_index=True
        )
        self.index_name = index_name

    def index_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        docs = [Document(content=row["text"].strip()) for _, row in df.iterrows() if isinstance(row["text"], str)]
        self.document_store.write_documents(docs)
