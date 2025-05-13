import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# nvidia/nv-embedcode-7b-v1 ---> Mistralbased code optimized
# nvidia/llama-3.2-nv-embedqa-1b-v2 ---> Llama based QA optimized

import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

class EmbeddingModel:
    def __init__(self, model_name="nvidia/llama-3.2-nv-embedqa-1b-v2", api_key=None, truncate="NONE"):
        """
        Initializes the EmbeddingModel with the specified parameters.

        :param model_name: The name of the model to be used for embeddings.
        :param api_key: The API key for NVIDIA services; can also be set via environment variable.
        :param truncate: The truncation strategy for the embeddings.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.truncate = truncate
        self.embeddings_model = self.get_embeddings_model()

    def get_embeddings_model(self):
        """
        Creates and returns an NVIDIAEmbeddings object.

        :return: An instance of the NVIDIAEmbeddings model.
        """
        embeddings_model = NVIDIAEmbeddings(
            model=self.model_name,
            api_key=self.api_key,
            truncate=self.truncate,
        )
        return embeddings_model

    def get_embeddings(self, text):
        """
        Retrieves embeddings for the given text input.

        :param text: The input text for which embeddings are to be generated.
        :return: The embeddings corresponding to the input text.
        """
        if not self.embeddings_model:
            raise ValueError("Embeddings model is not initialized.")
        
        return self.embeddings_model.embed(text)


