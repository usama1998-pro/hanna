from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import io

load_dotenv()


# print(len(string))

corpus_file = "corpus_info.txt"
vec_name = "AgilityModelInfoVectorStore"

print(f"Reading File [{corpus_file}]...")
corpus = io.open(corpus_file, "r", encoding="utf-8").read()


size = 2000
print(f"Splitting Text [chunk-size={size}]...")
# Splitting document in to Text
text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=size, length_function=len)
document = text_splitter.split_text(corpus)

print("Calling OpenAI Embeddings...")
embed = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

print("Converting to vectors..", end="\n")
vecdb = Chroma.from_texts(document, embed, persist_directory=vec_name)

print("Done!")

