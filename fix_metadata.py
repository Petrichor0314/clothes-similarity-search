import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_metadata():
    try:
        # Load existing embeddings to get IDs
        logger.info("Loading embeddings.json...")
        with open('embeddings.json', 'r') as f:
            embeddings_data = json.load(f)
        
        # Create ID set for quick lookup
        valid_ids = {item['id'] for item in embeddings_data}
        logger.info(f"Found {len(valid_ids)} valid IDs in embeddings.json")
        
        # Load current metadata
        logger.info("Loading current metadata.json...")
        with open('metadata.json', 'r') as f:
            old_metadata = json.load(f)
        
        # Create new metadata dictionary
        new_metadata = {}
        
        # Process each metadata entry
        logger.info("Processing metadata...")
        for item in old_metadata:
            try:
                # Generate the same ID that was used in embeddings
                product_name = item.get('name', '')
                link = item.get('link', '')
                
                # Extract front image URL
                product_images = eval(item.get('product_images', '[]'))
                front_image_url = None
                for image_dict in product_images:
                    for url in image_dict:
                        if '_1_1_1.jpg' in url:
                            front_image_url = url
                            break
                    if front_image_url:
                        break
                
                if not front_image_url:
                    logger.warning(f"No front image found for product: {product_name}")
                    continue
                
                # Create metadata entry
                metadata_entry = {
                    'name': product_name,
                    'price': item.get('price', 0),
                    'details': item.get('details', ''),
                    'link': link,
                    'gender': item.get('gender', ''),
                    'category': item.get('category', ''),
                    'image_url': front_image_url
                }
                
                # Only include items that have corresponding embeddings
                from hashlib import md5
                item_id = md5(f"{product_name}_{link}".encode()).hexdigest()
                
                if item_id in valid_ids:
                    new_metadata[item_id] = metadata_entry
                else:
                    logger.warning(f"ID {item_id} not found in embeddings")
            
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(new_metadata)} items")
        
        # Save new metadata
        logger.info("Saving new metadata.json...")
        with open('metadata.json', 'w') as f:
            json.dump(new_metadata, f)
        
        logger.info("Metadata fix completed successfully!")
        logger.info(f"Total items in new metadata: {len(new_metadata)}")
        
    except Exception as e:
        logger.error(f"Error fixing metadata: {str(e)}")
        raise

if __name__ == "__main__":
    fix_metadata()
