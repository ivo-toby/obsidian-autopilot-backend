"""Service for generating embeddings using various embedding models through LangChain."""

import logging
from typing import Any, Dict, List, Optional

from langchain_community.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    CohereEmbeddings,
    HuggingFaceInstructEmbeddings,
    OllamaEmbeddings,
)
from langchain_core.embeddings import Embeddings
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages the generation of embeddings for text content using various models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding service.

        Args:
            config: Configuration dictionary containing API settings and model preferences
        """
        self.config = config
        self.embedding_config = config.get("embeddings", {})
        self.model_type = self.embedding_config.get("model_type", "openai")
        self.model_name = self.embedding_config.get("model_name", "text-embedding-3-small")
        self.batch_size = self.embedding_config.get("batch_size", 100)

        # Initialize the embedding model based on configuration
        self.model = self._initialize_embedding_model()
        logger.info(f"Initialized embedding model: {self.model_type} - {self.model_name}")

    def _initialize_embedding_model(self) -> Embeddings:
        """Initialize the appropriate embedding model based on configuration."""
        if self.model_type == "openai":
            # Get API key from embeddings config, inference config, or legacy config
            api_key = (
                self.embedding_config.get("api_key") or
                self.config.get("inference", {}).get("openai", {}).get("api_key") or
                self.config.get("api_key", "")
            )
            return OpenAIEmbeddings(
                openai_api_key=api_key,
                model=self.model_name,
                chunk_size=self.batch_size,
            )
        elif self.model_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=self.embedding_config.get("model_kwargs", {}),
            )
        elif self.model_type == "huggingface_instruct":
            return HuggingFaceInstructEmbeddings(
                model_name=self.model_name,
                model_kwargs=self.embedding_config.get("model_kwargs", {}),
            )
        elif self.model_type == "cohere":
            return CohereEmbeddings(
                cohere_api_key=self.embedding_config.get("api_key"),
                model=self.model_name,
            )
        elif self.model_type == "ollama":
            ollama_config = self.embedding_config.get("ollama_config", {})
            return OllamaEmbeddings(
                model=self.model_name,
                base_url=ollama_config.get("base_url", "http://localhost:11434"),
                num_ctx=ollama_config.get("num_ctx", 512),
                num_thread=ollama_config.get("num_thread", 4),
                show_progress=True,
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {self.model_type}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single piece of text.

        Args:
            text: Text to generate embedding for

        Returns:
            List of embedding values
        """
        try:
            logger.debug("Generating embedding for query text")
            embeddings = self.model.embed_query(text)
            # Normalize embeddings
            if self.model_type == "openai":
                import numpy as np
                embeddings = np.array(embeddings)
                original_norm = np.linalg.norm(embeddings)
                logger.debug(f"Original embedding norm before normalization: {original_norm}")
                embeddings = embeddings / original_norm
                normalized_norm = np.linalg.norm(embeddings)
                logger.debug(f"Embedding norm after normalization: {normalized_norm}")
                embeddings = embeddings.tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def embed_chunks(
        self, chunks: List[str], show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple text chunks.

        Args:
            chunks: List of text chunks to generate embeddings for
            show_progress: Whether to show a progress bar

        Returns:
            List of embedding vectors
        """
        try:
            # Process in batches to avoid memory issues
            all_embeddings = []

            # For Ollama, we don't need to batch as it handles batching internally
            if self.model_type == "ollama":
                return self.model.embed_documents(chunks)

            # For other models, use our batching logic
            for i in tqdm(
                range(0, len(chunks), self.batch_size),
                disable=not show_progress,
                desc="Generating embeddings",
            ):
                batch = chunks[i : i + self.batch_size]
                batch_embeddings = self.model.embed_documents(batch)
                # Normalize OpenAI embeddings
                if self.model_type == "openai":
                    import numpy as np
                    batch_embeddings = [
                        (np.array(emb) / np.linalg.norm(np.array(emb))).tolist()
                        for emb in batch_embeddings
                    ]
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {str(e)}")
            raise

    def get_embedding_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current embedding configuration.

        Returns:
            Dictionary containing embedding metadata
        """
        metadata = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "dimensions": self._get_embedding_dimensions(),
            "normalized": True,
            "batch_size": self.batch_size,
        }

        # Add Ollama-specific metadata if using Ollama
        if self.model_type == "ollama":
            ollama_config = self.embedding_config.get("ollama_config", {})
            metadata.update({
                "ollama_base_url": ollama_config.get("base_url", "http://localhost:11434"),
                "ollama_num_ctx": ollama_config.get("num_ctx", 512),
                "ollama_num_thread": ollama_config.get("num_thread", 4),
            })

        return metadata

    def _get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for the current embedding model."""
        # Test with a simple string to get dimensions
        test_embedding = self.embed_text("test")
        return len(test_embedding)
