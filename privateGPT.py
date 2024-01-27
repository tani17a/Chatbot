import argparse
import os
import time
from constants import CHROMA_SETTINGS
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain.chains import RetrievalQA
from langchain.llms import GPT4All, LlamaCpp  
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#check if load_dotnet exists or not
if not load_dotenv():
    print("Could not load .env file or it is empty. Please check .env file is readable or not.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

#parse the command line arguments
#flags to disable enable printing of source document and streaming stdout callbacks
def parse_argument():
    parser=argparse.ArgumentParser(description='PrivateGPT: Provides the ansers of your questions accordings to the source documents. Makes eazy to talk with your documents.')
    parser.add_argument("--hide_source","-H",action='store_true',help="Use this flag to diable the printing of source documents of the answers provided by chatbot.")
    parser.add_argument("--mute_stream","-S",action='store_true',help="Use this flag to disable the streaming StdOut callback for LLMs.")
    return parser.parse_args()

def main():
    args=parse_argument()
    embeddings=HuggingFaceBgeEmbeddings(model_name=embeddings_model_name)
    chroma_client=chromadb.PersistentClient(path=persist_directory,settings=CHROMA_SETTINGS)
    db=Chroma(embedding_function=embeddings,persist_directory=persist_directory,client=chroma_client,client_settings=CHROMA_SETTINGS)
    #create a retriver object from initialized Chromadb object
    retriver=db.as_retriever(search_kwargs={"k":target_source_chunks})
    #set an object callback empty list if the mute_stream is trure else StreamingStdOutCallbackHandler
    callback=[] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    match model_type:
        case "GPT4All":
            llm=GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callback, verbose=False) 
        case "LlamaCpp":
            llm=LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callback, verbose=False)
        case _default:
            raise Exception(f"Model Type {model_type} is not supported. Please choose one of the following: LlamaCpp or GPT4All")
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriver,return_source_documents=not args.hide_source)
    while True:
        ques=input("\n Ask your question:")
        if ques=="exit":
            break
        if ques.strip()=="":
            continue

        start=time.time()
        result=qa(ques)
        answer,docs=result["result"],[] if args.hide_source else result["source_documents"]
        end=time.time()
        
        #print question and its answer
        print(" \n\n > Question:") 
        print(ques) 
        print(f"\n> Answer (took {round(end-start,2)}s.):") 
        print(answer)

        #Get the source of the answer
        for document in docs: 
            print("\n>"+document.metadata["source"]+":") 
            print(document.page_content)

if __name__=="__main__":
    main()
