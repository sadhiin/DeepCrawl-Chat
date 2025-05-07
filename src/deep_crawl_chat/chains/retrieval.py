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

def create_chat_chain():
    return get_rag_prompt()