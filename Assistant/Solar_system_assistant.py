"""
This module has some functionalities regarding text processing.

Author: Carlos Andrés Sierra <cavirguezs@udistrital.edu.co>
"""
import tkinter as tk
from tkinter import scrolledtext
import sys
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf_data(path: str) -> str:
    """
    This method takes all PDF files in a directory and loads them into a text string.

    Args:
        path (str): The path to the directory containing the PDF files.

    Returns:
        A text string containing the content of the PDF files.
    """
    loader = PyPDFDirectoryLoader(path)
    return loader.load()


def split_chunks(data: str) -> list:
    """
    This method splits a text string into chunks of 10000 characters
    with an overlap of 20 characters.

    Args:
        data (str): The text string to split into chunks.

    Returns:
        A list of strings containing the chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    chunks = splitter.split_documents(data)
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def get_embeddings() -> list:
    """
    This method gets semantic embeddings for a list of text chunks.

    Returns:
        A list of embeddings for the text chunks.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_chunk_embeddings(chunks: list, embeddings: list) -> list:
    """
    This method gets the embeddings for a list of text chunks.

    Args:
        chunks (list): A list of text chunks.
        embeddings (list): A list of embeddings for the text chunks.

    Returns:
        A list of embeddings for the text chunks.
    """
    return FAISS.from_documents(chunks, embedding=embeddings)


def load_llm():
    """
    This method loads the LLM model, in this case, one of the FOSS Mistral family.
    """
    # Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q3_K_M.gguf
    llm = LlamaCpp(
        streaming=True,
        model_path="c:/Users/Carlos Riveros/Documents/Marlon Riveros UD/Analisis de sistemas/mistral-7b-instruct-v0.2.Q3_K_M.gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096,
        force_download = True,
    )
    return llm


def agent_answer(question: str, llm: object, vector_store: object):
    """
    This method gets the answer to a question from the LLM model.

    Args:
        question (str): The question to ask the LLM model.
        llm (object): The LLM model.

    Returns:
        A string with the answer to the question.
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    )
    return qa.run(question)


def initialize_assistant():
    global llm, vector_store
    llm = load_llm()
    chunks = split_chunks(load_pdf_data("PDF"))
    embeddings = get_embeddings()
    vector_store = get_chunk_embeddings(chunks, embeddings)




def handle_question():
    question = question_entry.get()
    if question.strip():
        answer = agent_answer(question, llm, vector_store)
        conversation_text.insert(tk.END, f"User: {question}\n")
        conversation_text.insert(tk.END, f"Assistant: {answer}\n\n")
        question_entry.delete(0, tk.END)

# Crear la interfaz gráfica con tkinter
root = tk.Tk()
root.title("Solar System Assistant")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

conversation_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=50, height=20)
conversation_text.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

question_label = tk.Label(frame, text="Your question:")
question_label.grid(row=1, column=0, padx=5, pady=5)

question_entry = tk.Entry(frame, width=40)
question_entry.grid(row=1, column=1, padx=5, pady=5)

ask_button = tk.Button(frame, text="Ask", command=handle_question)
ask_button.grid(row=2, column=0, columnspan=2, pady=10)

exit_button = tk.Button(frame, text="Exit", command=sys.exit)
exit_button.grid(row=3, column=0, columnspan=2, pady=10)

initialize_assistant()

root.mainloop()