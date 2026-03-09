from langchain_core.prompts import ChatPromptTemplate


def get_rag_prompt():
    """Get a better RAG prompt with citation instructions"""
    return ChatPromptTemplate.from_template(
        """You are a helpful AI assistant that answers questions based only on the provided context.

        Context:
        {context}

        Guidelines:
        - Answer only based on the context provided
        - If the context doesn't contain the answer, say "I don't have enough information to answer this question"
        - Be concise but thorough
        - If appropriate, include relevant citations to the sources

        User Question: {input}
        """
    )

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def create_chat_chain(vectorstore):
    """Create a retrieval chain using a provided vectorstore."""
    from src.deepcrawl_chat.llm.models import get_llm
    
    llm = get_llm()
    prompt = get_rag_prompt()
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain