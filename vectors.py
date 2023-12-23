from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import io

load_dotenv()


string = """*Model Name: Three Universal Outcomes
Components: Always Ready, Always Responsive, and Always Innovative outcomes
Use: The Three Universal Outcomes guide structuring interactions that enable sustainability for the customers, organization, and workforce regardless of the future.
Additional Information:
URL_FOR_THIS_MODEL: https://enterpriseagility.community/three-universal-outcomes-of-enterprise-agility-10fx9k4s11
Autor: Erich R. Bühler and Enterprise Agility University
*End of Model Description"""


# print(len(string))

corpus_file = "corpus.txt"
vec_name = "AgilityModelVectorStore"

print(f"Reading File [{corpus_file}]...")
corpus = io.open(corpus_file, "r", encoding="utf-8").read()


size = 700
print(f"Splitting Text [chunk-size={size}]...")
# Splitting document in to Text
text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=size, length_function=len)
document = text_splitter.split_text(corpus)

print("Calling OpenAI Embeddings...")
embed = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

print("Converting to vectors..", end="\n")
vecdb = Chroma.from_texts(document, embed, persist_directory=vec_name)

print("Done!")

