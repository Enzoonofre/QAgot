import streamlit as st
from PIL import Image
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from transformers import pipeline

# Criando o objeto documentStore
document_store = InMemoryDocumentStore(use_bm25=True)

# Exportando os dados necessários sobre o Game Of Thrones
doc_dir = "data/build_your_first_question_answering_system"
fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
    output_dir=doc_dir,
)

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

retriever = BM25Retriever(document_store=document_store)

# Utilizando um pipeline da biblioteca Transformers
pipe = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

image = Image.open('comida.jpg')
st.image(image)
st.text("""QASports é um sistema de pergunta e respostas, o primeiro grande conjunto de dados
e respotas a perguntas de vários domínios sobre esportes para perguntas abertas""")

st.subheader('QASports',divider='rainbow')

user_input = None
if not user_input:
    user_input = st.text_input("Por favor, digite uma pergunta.")

if user_input:
    res = retriever.retrieve(user_input, top_k=5)  # Recupera os top 5 documentos relevantes
    if res:
        st.write(f"Foram encontrados {len(res)} documentos relevantes.")
        for document in res:
            prediction = pipe(question=user_input, context=document.content)
            context = document.content
            confidence = prediction["score"]
            answer = prediction["answer"]
            st.write("Pergunta:", user_input)
            st.write("Resposta:", answer)
            st.write("Confiança:", confidence)
            st.write("Contexto:", context)
            st.write("-" * 50)

if st.button('Buscar Resposta'):
    st.write(prediction["answer"])

