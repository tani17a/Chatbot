import os
import glob
from tqdm import tqdm
from typing import List
import chromadb 
from chromadb.config import Settings
from multiprocessing import Pool
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS

from langchain.document_loaders import(
    CSVLoader, 
    EverNoteLoader, 
    TextLoader, 
    UnstructuredEmailLoader, 
    UnstructuredHTMLLoader, 
    UnstructuredMarkdownLoader  , 
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader, 
    UnstructuredWordDocumentLoader, 
    UnstructuredEPubLoader, 
    UnstructuredHTMLLoader, 
    UnstructuredMarkdownLoader, 
    PyPDFLoader 
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings

if not load_dotenv():
  print("Could not load .env .Check if it exits and readable")
  exit(1)


chunk_size=1000 
chunk_overlap=100 
persist_directory=os.environ.get('PERSIST_DIRECTORY') 
source_directory=os.environ.get('SOURCE_DIRECTORY','source_documents') 
embeddings_model_name=os.environ.get('EMBEDDINGS_MODEL_NAME')

loader_mapping={
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

#To load single documnet
def Load_document(File_Path:str)->List[Document]:
    ext="."+File_Path.rsplit(".",-1)[1]
    if ext in loader_mapping:
        loader_class,loader_args=loader_mapping[ext]
        loader=loader_class(File_Path,**loader_args)
        loaded=loader.load()
        return loaded
    raise ValueError(f"Unsupported file extention: {ext}")

def load_multiple_document(souce_diectory:str,ignored_files:List[str]=[])->List[Document]:
    all_files=[]
    for ext in loader_mapping:
        all_files.extend(
            glob.glob(os.path.join(source_directory, f"**/*{ext.lower()}"), recursive=True))
        all_files.extend(
            glob.glob(os.path.join(source_directory, f"**/*{ext.upper()}"), recursive=True))
    filtered_file=[file_path for file_path in all_files if file_path not in ignored_files] 
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_file), desc='Loading new documents', ncols=80) as pbar:
         for i, docs in enumerate(pool.imap_unordered(Load_document, filtered_file)):
                results.extend(docs)
                pbar.update()
    return results


#Processes the files that are present in the source directory ignoring the already processed
def process_data(ignored_files:List[str]=[])->List[Document]:
    print(f"loading documents from {source_directory}")
    documents=load_multiple_document(source_directory,ignored_files)
    if not documents:
        print("No new documnets to load!!")
        exit(0)
    print(f"{len(documents)} are to be loaded from {source_directory}")
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    text=text_splitter.split_documents(documents)
    print(f"Split the {len(text)} in maxinmun chunk size: {chunk_size} each")
    return text


# function to embeddings and vector_space of the chunks
def vector_store_exits(persist_directory: str, embeddings: HuggingFaceBgeEmbeddings) -> bool:
    """
    Checks if vectorstore exists
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not db.get()['documents']:
        return False
    return True


# main function for storing the data 
def main():
    embeddings=HuggingFaceBgeEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    if vector_store_exits(persist_directory, embeddings):
        print("vectre store found")
        print(f"Appending the existing vectore space {persist_directory}....")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client=chroma_client)
        collection=db.get()
        source=[]
        text=process_data([metadata['source'] for metadata in collection['metadatas']])
        db.add_documents(text)
    else:
        print("Creating new vector space")
        print("Creating embedding!! it may take some time")
        text=process_data()
        db = Chroma.from_documents(text,embeddings,persist_directory=persist_directory,client=chroma_client)
    db.persist()
    db=None
    print(f"Ingestion complete! You can now run PrivateGPT.py to query your documents")


if __name__=="__main__":
    main()