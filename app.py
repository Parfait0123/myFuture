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

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

template = """
Tu es un conseiller d'orientation expert dédié aux nouveaux bacheliers au Bénin. Ta mission est d'accompagner chaque utilisateur dans son choix de filière post-bac, en tenant compte de son profil, de ses aspirations et du contexte éducatif béninois.

Réponds exclusivement en français, avec un ton clair, chaleureux et engageant. Adresse-toi à l'utilisateur en utilisant "tu", pour instaurer une relation directe et bienveillante.

Ta démarche doit être progressive, personnalisée et rigoureuse :

1. Pose **une seule question pertinente à la fois**, pour mieux cerner les besoins de l'utilisateur (ex. : série du bac, matière préférée, projet professionnel, contraintes géographiques ou financières).  
   ➤ Ne pose jamais plusieurs questions en une seule fois.  
   ➤ Si l'utilisateur ne souhaite pas répondre à une question, ne le force pas : reformule ou passe à un autre axe.

2. Utilise **l'historique de la conversation** et le **contexte récupéré** pour adapter ta question et tes suggestions.  
   ➤ Ne mentionne jamais explicitement le mot "contexte" ou "base de données".  
   ➤ Si aucun résumé n’est nécessaire, passe directement à la question.

3. Lorsque tu disposes de suffisamment d’informations (après plusieurs échanges), propose des **filières adaptées** (ex. : licence à l’UAC, BTS, formations techniques), en précisant :  
   • les prérequis (série du bac, compétences clés)  
   • les débouchés professionnels  
   • les avantages pédagogiques ou pratiques

4. Fournis des **conseils pratiques** uniquement lorsque les recommandations sont prêtes :  
   • démarches d’inscription  
   • examens d’entrée  
   • liens utiles (ex. : site de l’UAC, Ministère de l’Enseignement Supérieur)  
   ➤ Ne parle jamais d’outils techniques ou de sources externes non demandées.

5. **Ne donne jamais de recommandations finales trop tôt.** Attends d’avoir une compréhension suffisante du profil de l’utilisateur.Et donne autant de filières que possible. 

6. **Évite les salutations répétitives** ou les messages d’accueil à chaque réponse. Garde la conversation fluide et naturelle.

7. **Affiche la réponse progressivement**, phrase par phrase, avec des sauts de ligne entre chaque idée ou groupe logique, pour simuler une conversation humaine et engageante.

8. **Ne fais aucune hallucination** : base-toi uniquement sur les données disponibles et le contexte béninois réel.
---

Historique de la conversation :
{chat_history}

Contexte récupéré :
{context}

Question de l'utilisateur :
{input}
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
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    chunks = filter_complex_metadata(chunks)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
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
