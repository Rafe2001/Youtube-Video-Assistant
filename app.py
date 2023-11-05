from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate    
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

#video_url = "https://youtu.be/N9P5XiK55xM?si=OynmNUi8lR-zkv_f"

def create_db_from_yt_vid_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    splitting = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 100
    )
    
    docs = splitting.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db

#print(create_db_from_yt_vid_url(video_url))


def get_response_from_query(db, query, k=4):
    """
    davinci can handle max upto 4097 tokens.Setting chunksize to 1000 and k to 4 maximizes the number of tokens to analyze.
    """
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    llm = OpenAI(model_name = "text-davinci-003")
    
    prompt = PromptTemplate(
        input_variables = ["question", "docs"],
        template="""
        you are an helpful assistant that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you do not have enough information to answer the question, then say 'I don't know'.
        
        your answer should be verbose and detailed.
        """
    )
    
    chain = LLMChain(
        llm = llm,
        prompt=prompt
    )
    
    response  = chain.run(question = query, docs = docs_page_content)
    response = response.replace("/n","")
    return response, docs