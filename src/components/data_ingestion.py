from src.entity.artifact_entity import DataIngestionArtifact

from pathlib import Path
import os

import datasets

class DataIngestion:
    def __init__(self) -> DataIngestionArtifact:
        """
        Initializes the DataIngestion instance to copy data from HuggingFace Hub and save in Directory data in parquet format

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
            nested_json_input (dict): Input JSON data for ingestion.
        """
        pass

    def import_data_from_hub(self)->DataIngestionArtifact:
        try:
            data_dir = Path("Data")
            data_path = data_dir / "data.parquet"
            
            # Check if data already exists
            if data_path.exists():
                print(f"Data already exists at {data_path}")
            else:
                edgar_corpus_2020_raw = datasets.load_dataset("eloukas/edgar-corpus", "year_2020")
                os.makedirs(data_dir, exist_ok=True)
                edgar_corpus_2020_raw["train"].to_parquet(data_path)
                print(f"Data saved to {data_path}")

            # Create and return the data ingestion artifact
            data_ingestion_artifact = DataIngestionArtifact(corpus_file_path=str(data_path))
            return data_ingestion_artifact
            
        except Exception as e:
            raise(e)
        
                    


