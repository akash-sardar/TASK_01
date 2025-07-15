"""
Main Application Pipeline:
1. Data ingestion from HuggingFace datasets
2. Text chunking for embedding model
3. Embedding generation using sentence transformers
4. Clustering analysis with outlier detection and visualization

Output:
- Three visualization plots (clusters, outliers, sections)
- Summary statistics CSV file
- Results saved in 'Results/' directory
"""

from src.components.data_ingestion import DataIngestion
from src.components.chunking import Chunking
from src.components.embeddings import Embedding
from src.components.clustering import Clustering

try:
    # Step 1: Data Ingestion
    # Download SEC filing data from HuggingFace hub
    print("Starting data ingestion...")
    data_ingestion = DataIngestion()
    data_ingestion_artifact = data_ingestion.import_data_from_hub()
    print("Data ingestion completed.")

    # Step 2: Text Chunking
    # Split long documents into smaller chunks for embedding processing
    print("Starting text chunking...")
    chunking = Chunking(data_ingestion_artifact=data_ingestion_artifact)
    chunking_artifact = chunking.custom_text_splitter()
    print("Text chunking completed.")

    # Step 3: Embedding Generation
    # Convert text chunks to numerical embeddings using sentence transformers
    print("Starting embedding generation...")
    embedding = Embedding(chunking_artifact=chunking_artifact)
    embedding_artifact = embedding.embed_chunks()
    print("Embedding generation completed.")

    # Step 4: Clustering Analysis
    # Perform dimensionality reduction, clustering, and outlier detection
    print("Starting clustering analysis...")
    clustering = Clustering(embedding_artifact=embedding_artifact)
    clustering.perform_clustering()
    print("Clustering analysis completed.")
    
    print("\nPipeline completed successfully!")
    print("Check 'Results/' directory for visualization plots and summary statistics.")
    
except Exception as e:
    print(f"Pipeline failed with error: {str(e)}")
    raise(e)