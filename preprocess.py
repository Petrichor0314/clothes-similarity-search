import os
import json
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATASET_DIR = "dataset"
BATCH_SIZE = 32
MAX_WORKERS = 4

def setup_model():
    logger.info("Setting up model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load base model
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Use the full model except the last layer
    model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
    model = model.to(device)
    model.eval()
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Consistent size
        transforms.CenterCrop(224),     # Standard ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return model, transform, device

def extract_features(img_tensor, model):
    try:
        with torch.no_grad():
            features = model(img_tensor)
            features = features.cpu().squeeze().numpy()
            
            # Normalize features
            features = features / (np.linalg.norm(features) + 1e-8)
            
            logger.info(f"Extracted features shape: {features.shape}")
            return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None

def download_image(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        return None

def get_front_image(image_data):
    for image_dict in image_data:
        for url in image_dict:
            if '_1_1_1.jpg' in url:
                return url
    return None

def generate_unique_id(product_name, link):
    unique_string = f"{product_name}_{link}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def process_batch(batch_data, model, transform, device):
    logger.info(f"Processing batch of size: {len(batch_data)}")
    results = []
    for index, row in batch_data.iterrows():
        try:
            unique_id = generate_unique_id(row['Product_Name'], row['Link'])
            logger.info(f"Processing item with unique_id: {unique_id}")
            image_data = eval(row['Product_Image'])
            front_image_url = get_front_image(image_data)
            if not front_image_url:
                logger.warning(f"No front image found for unique_id: {unique_id}")
                continue
            image = download_image(front_image_url)
            if image is None:
                logger.warning(f"Failed to download image for unique_id: {unique_id}")
                continue
            img_tensor = transform(image).unsqueeze(0).to(device)
            embedding = extract_features(img_tensor, model)
            if embedding is None:
                logger.warning(f"Failed to extract features for unique_id: {unique_id}")
                continue
            results.append({
                'id': unique_id,
                'embedding': embedding,
                'metadata': {
                    'name': row['Product_Name'],
                    'price': row['Price'],
                    'details': row['Details'],
                    'link': row['Link'],
                    'gender': row['gender'],
                    'category': row['category'],
                    'product_images': row['Product_Image']
                }
            })
        except Exception as e:
            logger.error(f"Error processing unique_id {unique_id}: {e}")
    logger.info(f"Batch processing complete. Total successful items: {len(results)}")
    return results

def process_csv_file(csv_path, gender, category):
    logger.info(f"Processing CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path, header=0)
        df.columns = ['unnamed_id', 'Product_Name', 'Link', 'Product_Image', 'Price', 'Details']
        logger.info(f"Loaded {len(df)} records from {csv_path}")
        
        # Check if DataFrame is empty
        if df.empty:
            logger.warning(f"No data found in {csv_path}.")
            return pd.DataFrame()

        # Clean and process data
        df['gender'] = gender
        df['category'] = category
        df['Price'] = df['Price'].str.replace('₹', '').str.replace(',', '').astype(float)
        logger.info(f"Processing {len(df)} records for gender: {gender}, category: {category}")
        df = df[['Link', 'Product_Name', 'Price', 'Details', 'Product_Image', 'gender', 'category']]
        
        return df
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")
        return pd.DataFrame()

def main():
    # Initialize model
    model, transform, device = setup_model()
    
    # Process both Men's and Women's datasets
    all_data = []
    
    # Process both gender directories
    for gender in ['Men', 'Women']:
        gender_path = os.path.join(DATASET_DIR, gender, gender)
        if os.path.exists(gender_path):
            logger.info(f"Processing {gender} directory: {gender_path}")
            
            # Process each category file
            for csv_file in os.listdir(gender_path):
                if csv_file.endswith('.csv'):
                    category = os.path.splitext(csv_file)[0]  # Get category name from file name
                    csv_path = os.path.join(gender_path, csv_file)
                    df = process_csv_file(csv_path, gender, category)
                    if not df.empty:
                        all_data.append(df)
    
    # Combine all data
    if not all_data:
        logger.error("No data was processed. Check the dataset directory structure.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total items to process: {len(combined_df)}")
    
    # Process in batches
    all_results = []
    for i in tqdm(range(0, len(combined_df), BATCH_SIZE), desc="Processing batches"):
        batch = combined_df.iloc[i:i + BATCH_SIZE]
        results = process_batch(batch, model, transform, device)
        all_results.extend(results)
    
    # Separate embeddings and metadata
    embeddings_data = []
    metadata = {}  # Changed to dictionary
    
    for item in all_results:
        embeddings_data.append({
            'id': item['id'],
            'embedding': item['embedding'].tolist()  # Convert ndarray to list
        })
        metadata[item['id']] = {  # Use ID as key
            'name': item['metadata']['name'],
            'price': item['metadata']['price'],
            'details': item['metadata']['details'],
            'link': item['metadata']['link'],
            'gender': item['metadata']['gender'],
            'category': item['metadata']['category'],
            'image_url': get_front_image(eval(item['metadata']['product_images']))  # Store front image URL directly
        }
    
    logger.info(f"Successfully processed {len(all_results)} items")
    logger.info(f"Number of embeddings: {len(embeddings_data)}")
    logger.info(f"Number of metadata records: {len(metadata)}")
    
    # Verify all IDs match
    embedding_ids = set(item['id'] for item in embeddings_data)
    metadata_ids = set(metadata.keys())
    
    if embedding_ids != metadata_ids:
        logger.error("MISMATCH: Embedding IDs and metadata IDs don't match!")
        missing_in_metadata = embedding_ids - metadata_ids
        missing_in_embeddings = metadata_ids - embedding_ids
        if missing_in_metadata:
            logger.error(f"IDs in embeddings but missing in metadata: {missing_in_metadata}")
        if missing_in_embeddings:
            logger.error(f"IDs in metadata but missing in embeddings: {missing_in_embeddings}")
    else:
        logger.info("VERIFIED: All IDs match between embeddings and metadata")
    
    # Save embeddings and metadata
    logger.info("Saving embeddings and metadata...")
    with open('embeddings.json', 'w') as f:
        json.dump(embeddings_data, f)
    
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"Total processed items: {len(all_results)}")
    logger.info("Files saved: embeddings.json, metadata.json")

if __name__ == "__main__":
    main()
