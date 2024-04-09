import streamlit as st
from datasets import load_dataset
from PIL import Image
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from transformers import pipeline
from utils import get_unique_docs


# Carregar o dataset
dataset = load_dataset("PedroCJardim/QASports", "basketball")

# Inicializar o InMemoryDocumentStore
document_store = InMemoryDocumentStore(use_bm25=True)

unique_docs = set()
docs_validation = get_unique_docs(dataset["validation"], unique_docs)
docs_train = get_unique_docs(dataset["train"], unique_docs)
docs_test = get_unique_docs(dataset["test"], unique_docs)

document_store.write_documents(docs_validation) # Escrever os documentos de validação, para teste

# Executar o pipeline de indexação
# indexing_pipeline.run(docs=documents)
# Utilizando um pipeline da biblioteca Transformers
retriever = BM25Retriever(document_store=document_store)
pipe = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")


image = Image.open('assets/logo.png')
st.image(image)
st.markdown("""This website presents a collection of documents from the dataset named "QASports", the first large sports question answering dataset for open questions. QASports contains real data of players, teams and matches from the sports soccer, basketball and American football. It counts over 1.5 million questions and answers about 54k preprocessed, cleaned and organized documents from Wikipedia-like sources.""")
st.subheader('QASports: Basketball', divider='rainbow')

user_input = None
if not user_input:
    user_input = st.text_input("Please, make a question about basketball:")

if user_input:
    # Recupera os top 3 documentos relevantes
    res = retriever.retrieve(user_input, top_k=3)
    if res:
        st.write(f"It found {len(res)} relevant documents.")
        for document in res:
            prediction = pipe(question=user_input, context=document.content)
            context = document.content
            confidence = prediction["score"]
            answer = prediction["answer"]
            st.write("Question:", user_input)
            st.write("Answer:", answer)
            st.write("Confidence:", confidence)
            st.write("Context:", context)
            st.write("-" * 50)
