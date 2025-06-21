# src/ranker.py
from sentence_transformers import SentenceTransformer, util

class AnswerRanker:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def rank_answers(self, query, answers):
        if not answers:
            return None
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        best_score = -1
        best_answer = None
        for answer in answers:
            if not answer.answer:
                continue
            answer_embedding = self.model.encode(answer.answer, convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_embedding, answer_embedding).item()
            if score > best_score:
                best_score = score
                best_answer = answer
                best_answer.score = score  # Store score in answer object
        return best_answer
