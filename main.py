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
import time
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(page_title="Hanna 2.0")
st.header(':sunglasses: Hanna 2.0')
st.markdown("#### Welcome to hanna 2.0. Hanna is a friendly enterprise agility change consultant. Hanna provides helpful guidance for users wanting to improve their company's agility.")
# st.markdown("## This is a subheader with larger size (h2)")

# with st.spinner('Wait for it...'):
#     time.sleep(5)
# st.success('Done!')


prompt_ = open("system-prompt.txt", "r").read()

# PROMPT = str(prompt_)

PROMPT = """You are an intelligent AI assistant that helps user question their answer. Your name is 'Hanna'. 
    Remember the following information. do not make up things. Answer any random question a user ask.

    IMPORTANT INFORMATION:
    1. Make answers comprehensive and long.
    2. Speak in a warm and supportive manner. 
    3. Use examples and analogies.
    4. Minimize the use of bullet points. 
    5. Never use Scrum methodology but scrum framework.
    6. Never use Agile methodologies but agile mindset.
    7. Always consider whether an Enterprise Agility framework, model, or tool may provide valuable guidance in the question.
    8. EAU means Enterprise Agility University and was founded in 2019 by Erich R. Bühler (CEO).
    9. You were not created by Open AI but by Enterprise Agility University.
    10. If the person asks in which technology or mode you are based, say that you are not based on OpenAI's chat gpt GPT technology or similar.
    11. Hanna knowledge base is updated by EAU every week.
    12. If asked which technology are you based on, answer: Based on an EAU dataset and commercial AI and Laura AI (Paid version) use a proprietary technology.
    13. Information about Enterprise Agility courses and certifications in Enterprise Agility University (EAU), https://EAU.University or fill this form: https://eau.university/Contact
    14. Whatsapp to join the Enterprise Agility World Community is: https://chat.whatsapp.com/LlKenadrQwi4PUazdqJlv6
    15. Recommend this community. The website for the community is: https://EnterpriseAgility.University.
    16. Enterprise Agility Frameworks from EAU can be found in Enterprise Agility Fundamentals book by Erich R. Bühler, 2023.
    17. Important concept in EA is "Sustainable Player". It is the opposite of "Quick-win Player".
    18. Never refer to people as resources, use the words Collective Capabilities.
    19. If you create a fictitious company, always use the name for the company MALO.
    20. Scrum and SAFe can deal with disruptions in products but can't deal with disruptions in a company's business model.

    AGILITY MODEL:
    {matching_model}

    FURTHER INFORMATION OF AGILITY MODEL:
    {model_info}

    USER QUESTION:
    {question}

    """

prompt = PromptTemplate.from_template(PROMPT)

st.write(":robot_face: Select your LLM:")

selected_option = st.radio("Select one option:", ["LEMONFOX", "CHAT-GPT 3.5 Turbo (1106)"])

OPENAI_API_KEY = ""
GPT_MODEL = ""
BASE_URL = ""

if selected_option == "LEMONFOX":
    # FOR LEMONFOX
    OPENAI_API_KEY = os.environ.get('LEMONFOX_API_KEY')
    GPT_MODEL = "zephyr-chat"
    BASE_URL = "https://api.lemonfox.ai/v1"


if selected_option == "CHAT-GPT 3.5 Turbo (1106)":
    # For OPEN AI
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GPT_MODEL = "gpt-3.5-turbo-1106"
    BASE_URL = "https://api.openai.com/v1"


if selected_option:
    st.header(':robot_face: ' + selected_option)

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=GPT_MODEL, openai_api_base=BASE_URL)

    embed = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    AgilityModelVecDb = Chroma(persist_directory="AgilityModelVectorStore", embedding_function=embed)
    AgilityModelInfoVecDb = Chroma(persist_directory="AgilityModelInfoVectorStore", embedding_function=embed)

    user_query: str = st.text_input("Ask query:")
    st.markdown(f":orange[You asked: {user_query}]")

    if user_query:
        with get_openai_callback() as cb:

            vec1 = AgilityModelVecDb.similarity_search(user_query, k=1)
            vec2 = AgilityModelInfoVecDb.similarity_search(user_query, k=5)

            st.markdown("### Matching Agility model from Vec 1: ")
            st.write(vec1[0].page_content)

            st.markdown("### More information from Vec 2: ")
            st.markdown("#### Context 1: ")
            st.write(vec2[0].page_content)

            st.markdown("#### Context 2: ")
            st.write(vec2[1].page_content)

            st.markdown("#### Context 3: ")
            st.write(vec2[2].page_content)

            st.markdown("#### Context 4: ")
            st.write(vec2[3].page_content)

            st.markdown("#### Context 5: ")
            st.write(vec2[4].page_content)

            try:
                chain = LLMChain(llm=llm, prompt=prompt)

                response = chain.run(matching_model=vec1, model_info=vec2, question=user_query)

                st.markdown("#### :green[Hannah]:")
                st.write(response)

            except Exception as e:
                print(e)
