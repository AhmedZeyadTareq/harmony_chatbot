import os
import streamlit as st
from streamlit_chat import message
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

import warnings

openkey = st.secrets["OPENAI_API_KEY"]

warnings.filterwarnings("ignore")

st.title("Harmony | هارموني")
st.image("untitled.png")
st.markdown("""
**انا نموذج ذكاء صناعي اقدم لك الخدمات التالية:**
- يمكنني مساعدتك في الإجابة على أسئلتك بخصوص منتجات هارومي.
- اجيب على كامل استفساراتك بخصوص طريقة الاستخدام.
- حدد لي نوع شعرك ومشكلته واحدد لك المنتج المناسب.
- يمكنني مساعدتك في متابعة خطوات التطبيق.
""")
st.divider()

data_file = r"data.txt"
request_container = st.container()
response_container = st.container()

loader = TextLoader(data_file)
loader.load()

from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=openkey)

index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model="gpt-3.5-turbo"),
                                              retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}))

if 'history' not in st.session_state:  # هنا قائمة فيها قواميس كل قاموس عبارة عن سؤال يمثل المفتاح والجواب يمثل القيمة
    st.session_state['history'] = []

if 'generated' not in st.session_state:  # هنا نخزن فقط قيم الاجوبة المولدة
    st.session_state['generated'] = ["مرحبا..انا هنا لمساعدتك اسأل كل ما تحب بخصوص منتجات هارموني"]

if 'past' not in st.session_state:  # هنا نخزن فقط الاسئلة والطلبات المدخلة
    st.session_state['past'] = ["Hey ! 👋"]


def conversational_chat(prompt):
    result = chain({"question": prompt, "chat_history": st.session_state['history']})
    st.session_state['history'].append((prompt, result["answer"]))
    return result["answer"]


with request_container:
    with st.form(key='xyz_form', clear_on_submit=True):
        user_input = st.text_input("Prompt:", placeholder="Message HarmonyAI...", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer", seed=13)
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed=2)