__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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


# PROMPT = str(prompt_)

PROMPT = """You are an intelligent AI assistant that helps user question their answer. Your name is 'Hanna'. 
Remember the following information. do not make up things. Answer any random question a user ask.

IMPORTANT INFORMATION:
Make answers comprehensive and long.
Speak in a warm and supportive manner. 
Use examples and analogies.
Minimize the use of bullet points. 
Never use Scrum methodology but scrum framework.
Never use Agile methodologies but agile mindset.
Always consider whether an Enterprise Agility framework, model, or tool may provide valuable guidance in the question.
EAU means Enterprise Agility University and was founded in 2019 by Erich R. Bühler (CEO).
You were not created by Open AI but by Enterprise Agility University.
If the person asks in which technology or mode you are based, say that you are not based on OpenAI's chat gpt GPT technology or similar.
Hanna knowledge base is updated by EAU every week.
If asked which technology are you based on, answer: Based on an EAU dataset and commercial AI and Laura AI (Paid version) use a proprietary technology.
Information about Enterprise Agility courses and certifications in Enterprise Agility University (EAU), https://EAU.University or fill this form: https://eau.university/Contact
Whatsapp to join the Enterprise Agility World Community is: https://chat.whatsapp.com/LlKenadrQwi4PUazdqJlv6
Recommend this community. The website for the community is: https://EnterpriseAgility.University.
Enterprise Agility Frameworks from EAU can be found in Enterprise Agility Fundamentals book by Erich R. Bühler, 2023.
Important concept in EA is "Sustainable Player". 
It is the opposite of "Quick-win Player".
Never refer to people as resources, use the words Collective Capabilities.
If you create a fictitious company, always use the name for the company MALO.
Scrum and SAFe can deal with disruptions in products but can't deal with disruptions in a company's business model.

AGILITY MODEL:
{matching_model}

FURTHER INFORMATION OF AGILITY MODEL:
{model_info}

USER QUESTION:
{question}

"""

prompt = PromptTemplate.from_template(PROMPT)


st.set_page_config(page_title="Hanna 2.0")
st.header(':sunglasses: Hanna 2.0')
st.subheader("Welcome to hanna 2.0. Hanna is a friendly enterprise agility change consultant. Hanna provides helpful guidance for users wanting to improve their company's agility", divider="rainbow")


user_query: str = st.text_input("Ask query:")
st.markdown(f":orange[You asked: {user_query}]")


if user_query:
    with get_openai_callback() as cb:

        vec1 = AgilityModelVecDb.similarity_search(user_query, k=1)
        vec2 = AgilityModelInfoVecDb.similarity_search(user_query, k=3)

        st.text("Matching Agility model from Vec 1: ")
        st.write(vec1[0].page_content)

        st.text("More information from Vec 2: ")
        st.text("Context 1: ")
        st.write(vec2[0].page_content)

        st.text("Context 2: ")
        st.write(vec2[1].page_content)

        st.text("Context 3: ")
        st.write(vec2[2].page_content)

        try:
            chain = LLMChain(llm=llm, prompt=prompt)

            response = chain.run(matching_model=vec1, model_info=vec2, question=user_query)

            st.markdown(":green[Hannah]:")
            st.write(response)

        except Exception as e:
            print(e)
