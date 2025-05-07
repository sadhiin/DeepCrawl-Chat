import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# nvidia/nv-embedcode-7b-v1 ---> Mistralbased code optimized
# nvidia/llama-3.2-nv-embedqa-1b-v2 ---> Llama based QA optimized


def get_embeddings_model():
    embeddings_model = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="NONE", )

    return embeddings_model