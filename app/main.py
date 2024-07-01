import uvicorn
import logging
import os
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
logger.setLevel(logging.DEBUG)

load_dotenv('env')

# Database
# CONNECTION_URL = os.environ['connectionURL']

sentences = ["This is an example sentence", "Each sentence is converted"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=["Origin, X-Requested-With, Content-Type, Accept"],
)

class Body(BaseModel):
	text: str

@app.get('/')
def root():
	return Response('-- FASTAPI working --')

@app.post('/generate')
def predict(body: Body):
	# generated_output = generator(body.text, max_length=35, num_return_sequences=1)
	# output_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
	model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
	embeddings = model.encode(sentences)
	return embeddings


# if __name__ == "__main__":
#     uvicorn.run('main:app', host="0.0.0.0", reload=True, debug=True)