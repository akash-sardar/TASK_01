# --------------------------------------------------
# artifact_entity.py
# --------------------------------------------------
# This module defines data structures using Python's dataclass for 
# handling artifacts generated during the data ingestion and request ingestion stages.
# --------------------------------------------------

from dataclasses import dataclass
from pyspark.sql import DataFrame
import pandas as pd

# --------------------------------------------------
# Data Ingestion Artifact Class
# --------------------------------------------------
@dataclass
class DataIngestionArtifact:
    """
    DataIngestionArtifact stores information about the artifacts 
    generated during the data ingestion phase.
    
    Attributes:
        corpus_file_path (str): Path to the system file generated during ingestion.
    """
    corpus_file_path: str  #  Path to the ingested file

# --------------------------------------------------
# Chunking Artifact Class
# --------------------------------------------------
@dataclass
class ChunkingArtifact:
    """
    Chunks the dataset into pieces to be able to be ingested by Sentence Transformers
    
    Attributes:
        chunked_dataframe (DataFrame): Path to chunked data
    """
    chunked_dataframe: DataFrame  #  chunked dataframe   

# --------------------------------------------------
# Embedding Artifact Class
# --------------------------------------------------
@dataclass
class EmbeddingArtifact:
    """
    Converts the chunks into Embeddings
    
    Attributes:
        embedded_dataframe (DataFrame)
    """
    df: pd.DataFrame # pandas dataframe with embeddings included
    spark_df: DataFrame # Sparkdataframe with embeddings included

