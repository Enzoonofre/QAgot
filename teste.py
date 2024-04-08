import streamlit as st
from datasets import load_dataset
from PIL import Image
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from transformers import pipeline
import mmh3 

docs = list()
unique_docs = set()

# Carregar o dataset
dataset = load_dataset("PedroCJardim/QASports", "basketball")

# Inicializar o InMemoryDocumentStore
document_store = InMemoryDocumentStore(use_bm25=True)


for i in dataset["validation"]:
    if i["context"] is not None:
        document_id = mmh3.hash128(i["context"], signed=False)
        if document_id not in unique_docs:
            unique_docs.add(document_id)

            dicionario = {}
            dicionario['content'] = i['context']
            dicionario['id'] = document_id
        
            aux = {}
            aux['context_title'] = i['context_title']
            aux['context.id'] = i['context_id']
            dicionario['meta'] = aux

            docs.append(dicionario)
    
for i in dataset["train"]:
    if i["context"] is not None:
        document_id = mmh3.hash128(i["context"], signed=False)
        if document_id not in unique_docs:
            unique_docs.add(document_id)

            dicionario = {}
            dicionario['content'] = i['context']
            dicionario['id'] = document_id
        
            aux = {}
            aux['context_title'] = i['context_title']
            aux['context.id'] = i['context_id']
            dicionario['meta'] = aux

            docs.append(dicionario)
    

for i in dataset["test"]:
    if i["context"] is not None:
        document_id = mmh3.hash128(i["context"], signed=False)
        if document_id not in unique_docs:
            unique_docs.add(document_id)

            dicionario = {}
            dicionario['content'] = i['context']
            dicionario['id'] = document_id
        
            aux = {}
            aux['context_title'] = i['context_title']
            aux['context.id'] = i['context_id']
            dicionario['meta'] = aux

            docs.append(dicionario)




document_store.write_documents(docs)

# # Executar o pipeline de indexação
# indexing_pipeline.run(docs=documents)

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
    