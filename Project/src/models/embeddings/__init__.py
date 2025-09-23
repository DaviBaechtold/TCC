"""
Modelos de embeddings de vídeo.
"""

from .video_embeddings import (
    VideoEmbeddingExtractor,
    MotionEmbedding,
    MultiScaleVideoEmbedding
)

__all__ = [
    "VideoEmbeddingExtractor",
    "MotionEmbedding", 
    "MultiScaleVideoEmbedding"
]