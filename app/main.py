import uvicorn
import logging
import os
import torch
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from fastapi import FastAPI, Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# generator = pipeline('text-generation', model_name)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv('env')

# Database
# CONNECTION_URL = os.environ['connectionURL']

# sentences = ["This is an example sentence", "Each sentence is converted"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=["Origin, X-Requested-With, Content-Type, Accept"],
)

@app.get('/')
def root():
	return Response('-- FASTAPI working --')

@app.post('/suggestion')
def predict(query: str, sentences: List[str]):

	model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
	
	#encode query
	query_embedding = model.encode(query)

	#encode all available sentences from which we search query
	content_embeddings = model.encode(sentences)
	
	#it will score between query and sentences
	similarity_scores = model.similarity(query_embedding, content_embeddings)[0]

	#pick 5 top most scores
	top_k = min(5, len(content_embeddings))
	scores, indices = torch.topk(similarity_scores, k=top_k)

	output = []
	#for top 5 render sentence with score
	for score, idx in zip(scores, indices):
	  output.append(sentences[idx] + " - (Score: {:.4f})".format(score))

	return output


# if __name__ == "__main__":
#     uvicorn.run('main:app', host="0.0.0.0", reload=True, debug=True)