from src.entity.artifact_entity import ChunkingArtifact, EmbeddingArtifact

import torch
import numpy as np
from pyspark.sql import SparkSession

from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self, chunking_artifact: ChunkingArtifact) -> EmbeddingArtifact:
        """
        Performs embedding of chunked dataset using sentence transformer.
        
        Args:
            chunking_artifact: ChunkingArtifact containing PySpark DataFrame with 
                             exploded chunks as rows
        """
        try:
            self.chunking_artifact = chunking_artifact
            # Initialize Spark session for data processing
            self.spark = SparkSession.builder \
                .appName("AIG_SEC_Filing_Analysis") \
                .getOrCreate()            
        except Exception as e:
            raise(e)
        

    def convert_to_pandas(self, df):
        """
        Convert PySpark DataFrame to Pandas DataFrame for faster local processing.
        
        Args:
            df: PySpark DataFrame containing chunked data
            
        Returns:
            Pandas DataFrame with selected columns for embedding
            
        Note:
            Stops Spark session after conversion to free resources
        """
        try:
            # Select only required columns for embedding
            df_pandas = df.select("cik", "filename", "section_name", "chunk", "chunk_id").toPandas()
            
            # Stop Spark session to free resources
            # self.spark.stop()
            
            return df_pandas
        except Exception as e:
            raise(e)


    def embed_chunks(self):
        """
        Main method to embed chunked data using sentence transformers.
        
        Uses all-MiniLM-L6-v2 model for generating 384-dimensional embeddings.
        Processes chunks in batches for memory efficiency.
        
        Returns:
            EmbeddingArtifact containing pandas DataFrame with embeddings column
        """
        try:
            # Get chunked dataframe from artifact
            df_exploded = self.chunking_artifact.chunked_dataframe

            # Convert to pandas for faster local processing
            df_chunked_pandas = self.convert_to_pandas(df_exploded)

            # Use GPU if available, otherwise CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load sentence transformer model
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

            # Extract chunks as list for batch processing
            chunks_list = df_chunked_pandas["chunk"].tolist()
            
            # Generate embeddings in batches
            embeddings = model.encode(chunks_list, 
                                    batch_size=32,         
                                    show_progress_bar=True,  
                                    convert_to_tensor=False) 
            if np.isnan(embeddings).any():
                raise(f"embeddings have NaN values")
            
            empty_vectors = np.sum(embeddings, axis=1) == 0

            if np.sum(empty_vectors) > 0:
                raise(f"Embeddings have empty vectors")

            # Add embeddings as new column to dataframe
            df_chunked_pandas["embeddings"] = embeddings.tolist()

            # create Pyspark Dataframe for future usage
            df_cleaned = self.spark.createDataFrame(df_chunked_pandas)

            # Create and return embedding artifact
            embedding_artifact = EmbeddingArtifact(df=df_chunked_pandas, spark_df = df_cleaned)

            return embedding_artifact

        except Exception as e:
            raise(e)