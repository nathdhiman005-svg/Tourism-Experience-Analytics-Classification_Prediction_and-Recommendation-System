import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestor:
    """
    A class to handle the loading of raw data files based on the configuration.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(__file__).parent / config_path
        self.config = self._load_config()
        self.raw_data_paths = self.config.get('data_paths', {}).get('raw', {})

    def _load_config(self) -> Dict[str, Any]:
        """Loads the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Successfully loaded configuration from {self.config_path.name}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Loads all raw excel datasets specified in the configuration into a dictionary of DataFrames.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are dataset names and values are DataFrames.
        """
        datasets = {}
        for name, relative_path in self.raw_data_paths.items():
            # Resolve the path relative to the src directory where config.yaml is
            file_path = (self.config_path.parent / relative_path).resolve()
            
            try:
                logger.info(f"Loading {name} dataset from {file_path}")
                df = pd.read_excel(file_path)
                datasets[name] = df
                logger.info(f"Successfully loaded {name} dataset with shape {df.shape}")
            except FileNotFoundError:
                logger.error(f"Dataset file not found: {file_path}")
            except Exception as e:
                logger.error(f"Error loading dataset {name} from {file_path}: {e}")
                
        return datasets

    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges the raw relational datasets into a single wide feature store dataset.
        """
        logger.info("Starting relational merging logic...")
        
        # 1. Enriched User Data
        user = datasets['user'].copy()
        
        # Merge Continent
        user = user.merge(datasets['continent'], on='ContinentId', how='left')
        user.rename(columns={'Continent': 'UserContinent'}, inplace=True)
        
        # Merge Region
        region = datasets['region'].drop(columns=['ContinentId'], errors='ignore')
        user = user.merge(region, on='RegionId', how='left')
        user.rename(columns={'Region': 'UserRegion'}, inplace=True)
        
        # Merge Country
        country = datasets['country'].drop(columns=['RegionId'], errors='ignore')
        user = user.merge(country, on='CountryId', how='left')
        user.rename(columns={'Country': 'UserCountry'}, inplace=True)
        
        # Merge City
        city = datasets['city'].drop(columns=['CountryId'], errors='ignore')
        user = user.merge(city, on='CityId', how='left')
        user.rename(columns={'CityName': 'UserCity'}, inplace=True)
        
        # 2. Enriched Item (Attraction) Data
        item = datasets['item'].copy()
        
        # Merge Attraction Type
        item = item.merge(datasets['type'], on='AttractionTypeId', how='left')
        
        # Merge Attraction City
        item = item.merge(city, left_on='AttractionCityId', right_on='CityId', how='left')
        item.rename(columns={'CityName': 'AttractionCity'}, inplace=True)
        item.drop(columns=['CityId'], inplace=True, errors='ignore')
        
        # 3. Enriched Transaction Data
        transaction = datasets['transaction'].copy()
        
        # Merge Mode
        # The transaction dataset column is 'VisitMode' but it contains IDs
        transaction = transaction.merge(datasets['mode'], left_on='VisitMode', right_on='VisitModeId', how='left', suffixes=('_id', ''))
        transaction.rename(columns={'VisitMode': 'VisitModeName'}, inplace=True)
        
        # Merge with Enriched User
        merged_df = transaction.merge(user, on='UserId', how='left')
        
        # Merge with Enriched Item
        merged_df = merged_df.merge(item, on='AttractionId', how='left')
        
        logger.info(f"Merging complete. Final dataset shape: {merged_df.shape}")
        return merged_df

if __name__ == "__main__":
    # Execute Pipeline (Steps 1.2, 1.3, 1.4)
    ingestor = DataIngestor()
    datasets = ingestor.load_all_datasets()
    
    # Merge datasets (Step 1.3)
    merged_data = ingestor.merge_datasets(datasets)
    
    # Save the consolidated dataset (Step 1.4)
    out_path = (ingestor.config_path.parent / ingestor.config.get('data_paths', {}).get('processed', {}).get('raw_merged', '../processed_data/raw_merged.csv')).resolve()
    logger.info(f"Saving merged dataset to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_data.to_csv(out_path, index=False)
    logger.info("Pipeline execution finished successfully.")
