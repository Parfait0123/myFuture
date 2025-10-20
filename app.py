import os
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
import time

# Configurer la clé API (utilisez st.secrets en production)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", "AIzaSyAn9FxGvhG97YASL-XJiuwmpbm5vxtcCvg"))

# Prompt amélioré (ton ouvert, structuré, une question à la fois)
template = """
Tu es un conseiller d'orientation expert pour les nouveaux bacheliers au Bénin. Réponds en français, de manière claire, chaleureuse et engageante, en t'adressant directement à l'utilisateur avec "tu". Ton rôle est d'accompagner l'utilisateur dans son choix de filière en posant une seule question pertinente à la fois pour mieux comprendre son profil, ses intérêts et ses objectifs, avant de proposer des recommandations. Utilise l'historique de la conversation et le contexte pour personnaliser ta question et tes suggestions, en te concentrant sur le système éducatif béninois (ex. : universités comme UAC, UNSTIM, formations professionnelles). Ne donne pas de recommandations finales tant que tu n'as pas suffisamment d'informations sur l'utilisateur. Évite les salutations répétitives ou les messages d'accueil à chaque réponse pour garder la conversation fluide et naturelle.

Historique de la conversation :
{chat_history}

Contexte récupéré : {context}

Question de l'utilisateur : {input}

Suis ces étapes dans ta réponse :
1. Résume brièvement ce que tu as compris de son profil ou de sa question, en te basant sur l'historique et sa dernière entrée, sans mentionner de détails techniques comme le contexte ou la base de données.
2. Pose une seule question précise et pertinente pour approfondir la compréhension de ses besoins (ex. : série du bac, matière préférée, intérêt professionnel, contraintes financières ou géographiques). Évite d'encombrer avec plusieurs questions.
3. Si tu as assez d'informations (après plusieurs échanges), propose des filières adaptées (ex. : licence à l'UAC, formations techniques) avec leurs prérequis (série du bac, compétences clés) et débouchés professionnels.
4. Fournis des conseils pratiques (ex. : démarches d'inscription, examens d'entrée, ressources comme le site de l'UAC ou le Ministère de l'Enseignement Supérieur) uniquement lorsque les recommandations sont prêtes.
Concentre-toi uniquement sur mes besoins et évite de parler de choses non pertinentes comme des outils techniques ou des sources externes non demandées.

Affiche la réponse progressivement, phrase par phrase, avec des sauts de ligne pour chaque phrase ou groupe logique, pour simuler une conversation naturelle et engageante.
"""

# Charger les documents Excel (caché pour performances)
@st.cache_resource
def load_documents():
    documents = []
    folder_path = "./formations_traités"
    if not os.path.exists(folder_path):
        st.error("Dossier 'formations_traités' non trouvé. Veuillez uploader les fichiers Excel.")
        return documents
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            documents.extend(loader.load())
    return documents

# Initialiser la chaîne RAG (sans mémoire, car session-based)
@st.cache_resource
def init_chain():
    documents = load_documents()
    if not documents:
        return None
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    chunks = filter_complex_metadata(chunks)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    prompt = PromptTemplate.from_template(template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain

# Interface Streamlit
st.title("Assistant Choix de Filière au Bénin")
st.write("Posez vos questions sur votre orientation post-bac. L'assistant vous guidera étape par étape.")

# Uploader les fichiers Excel (optionnel)
uploaded_files = st.sidebar.file_uploader("Uploadez vos fichiers Excel (optionnel)", accept_multiple_files=True, type="xlsx")
if uploaded_files:
    os.makedirs("./formations_traités", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("./formations_traités", file.name), "wb") as f:
            f.write(file.getbuffer())

# Initialiser la chaîne RAG (cachée)
retrieval_chain = init_chain()

# Gérer la mémoire par session (clé pour résoudre le problème partagé)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="answer",
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique du chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if prompt := st.chat_input("Votre question (tapez 'quit' pour quitter) :"):
    if prompt.lower() == "quit":
        st.write("Au revoir ! Bonne chance pour votre orientation.")
        st.session_state.messages.append({"role": "assistant", "content": "Au revoir ! Bonne chance pour votre orientation."})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                if retrieval_chain:
                    try:
                        response = retrieval_chain.invoke({
                            "input": prompt, 
                            "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]
                        })
                        
                        # Streaming simulé
                        st.markdown("**Réponse de l'assistant :**")
                        sentences = response["answer"].split("\n")
                        for sentence in sentences:
                            if sentence.strip():
                                st.markdown(sentence)
                                time.sleep(0.3)
                        full_response = response["answer"]
                        
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.session_state.memory.save_context({"input": prompt}, {"answer": full_response})
                    except Exception as e:
                        st.error(f"Erreur : {e}. Vérifiez votre clé API ou les fichiers Excel.")
                else:
                    st.error("Aucun fichier Excel chargé. L'assistant ne peut pas répondre.")
else:
    if not retrieval_chain:
        st.info("Uploadez des fichiers Excel pour démarrer.")
