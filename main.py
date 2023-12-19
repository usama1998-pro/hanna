import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

GPT_MODEL = "gpt-3.5-turbo-1106"

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=GPT_MODEL)


embed = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))


AgilityModelVecDb = Chroma(persist_directory="AgilityModelVectorStore", embedding_function=embed)
AgilityModelInfoVecDb = Chroma(persist_directory="AgilityModelInfoVectorStore", embedding_function=embed)


prompt_ = open("system-prompt.txt", "r").read()


PROMPT = """Never make up the names or dimensions of the frameworks or models. If you don't know, don't mention the name of the dimensions and just mention the acronym.

For ***MODEL DEFINITION v1.0*** file, this is the following properties:

*Model Name:
Components:
Use:
Additional Information:
URL_FOR_THIS_MODEL:
Autor:
*End of Model Description


CONTEXT:
{input_documents}"""

prompt = PromptTemplate.from_template(PROMPT)


st.set_page_config(page_title="Hanna 2.0")
st.header(':sunglasses: Hanna 2.0')
st.subheader("Welcome to hanna 2.0. Hanna is a friendly enterprise agility change consultant. Hanna provides helpful guidance for users wanting to improve their company's agility", divider="rainbow")


user_query: str = st.text_input("Ask query:")
st.markdown(f":orange[You asked: {user_query}]")


if user_query:
    with get_openai_callback() as cb:

        vec1 = AgilityModelVecDb.similarity_search(user_query, k=3)
        vec2 = AgilityModelInfoVecDb.similarity_search(vec1[0].page_content, k=3)

        st.text("Found information from Vector 1 according to your query!")
        st.write(vec1[0].page_content)

        st.text("Here is further information from Vector 2 for Vector 1:")
        st.write(vec2[0].page_content)

        try:
            chain = LLMChain(llm=llm, prompt=prompt)

            final_vec = vec1[0].page_content + "\n\n" + vec2[0].page_content

            response = chain.run(input_documents=final_vec, question=user_query)

            st.markdown(":green[Hannah]:")
            st.write(response)

        except Exception as e:
            print(e)
