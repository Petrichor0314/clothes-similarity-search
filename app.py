import os
import json
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import logging
from flask_cors import CORS
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Enable debug mode
app.config['DEBUG'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for caching
embeddings = None
metadata_dict = None
model = None
transform = None
product_ids = []

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

def load_model():
    global model, transform
    try:
        logger.info("Loading ResNet50 model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load base model
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Use the full model except the last layer
        model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        model = model.to(device)
        model.eval()
        
        # Update transform to better handle fashion images
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Consistent size
            transforms.CenterCrop(224),     # Standard ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        logger.info("Model and transform loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def load_data():
    global embeddings, metadata_dict, product_ids
    try:
        logger.info("Loading embeddings from embeddings.json...")
        with open('embeddings.json', 'r') as f:
            embeddings_data = json.load(f)
            embeddings = np.array([item['embedding'] for item in embeddings_data]).astype('float32')
            logger.info("Normalizing embeddings...")
            norms = np.linalg.norm(embeddings, axis=1)
            embeddings = embeddings / norms[:, np.newaxis]
            product_ids = [item['id'] for item in embeddings_data]
        
        logger.info("Loading metadata from metadata.json...")
        with open('metadata.json', 'r') as f:
            metadata_dict = json.load(f)
        
        # Verify counts match
        logger.info(f"Loaded {len(embeddings)} embeddings and {len(metadata_dict)} metadata records")
        if len(embeddings) != len(metadata_dict):
            logger.error(f"MISMATCH: Number of embeddings ({len(embeddings)}) does not match number of metadata records ({len(metadata_dict)})")
            
        # Verify all IDs match
        embedding_ids = set(product_ids)
        metadata_ids = set(metadata_dict.keys())
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
        
        return True
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

def compute_distances(query_vector, k=50):
    """Compute cosine similarity between query vector and all embeddings"""
    # Ensure vectors are normalized
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    similarities = np.dot(embeddings_norm, query_norm)
    
    # Get top k most similar items
    indices = np.argsort(similarities)[-k:][::-1]
    sorted_similarities = similarities[indices]
    
    # Convert to distances (for consistency with previous code)
    sorted_distances = 1 - sorted_similarities
    
    return sorted_distances, indices

def extract_features(image):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        with torch.no_grad():
            print("Transforming image...")
            img_tensor = transform(image).unsqueeze(0)
            print("Image transformed successfully")
            
            print("Moving tensors to device...")
            img_tensor = img_tensor.to(device)
            print("Tensors moved to device successfully")
            
            print("Extracting features...")
            features = model(img_tensor)
            features = features.squeeze()
            print("Features extracted successfully")
            
            print("Moving features to CPU...")
            features_cpu = features.cpu().numpy()
            print("Features moved to CPU successfully")
            
            # Normalize features
            features_cpu = features_cpu / (np.linalg.norm(features_cpu) + 1e-8)
            
            print(f"Final feature shape: {features_cpu.shape}")
        return features_cpu
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search/image', methods=['POST'])
def search_by_image():
    print("\n=== New Image Search Request ===")
    try:
        # Check initialization
        print("Checking initialization state...")
        if not model:
            print("ERROR: Model not initialized")
            return jsonify({'error': 'Model not initialized'}), 500
        if embeddings is None:
            print("ERROR: Embeddings not loaded")
            return jsonify({'error': 'Embeddings not loaded'}), 500
        if not metadata_dict:
            print("ERROR: Metadata not loaded")
            return jsonify({'error': 'Metadata not loaded'}), 500
            
        print("All components initialized properly")
        
        # Check request
        print("Checking request files...")
        if 'image' not in request.files:
            print("ERROR: No image file in request")
            return jsonify({'error': 'No image uploaded'}), 400
        
        # Get parameters
        gender = request.form.get('gender', 'all')
        category = request.form.get('category', 'all')
        print(f"Search parameters - Gender: {gender}, Category: {category}")
        
        # Process image
        image_file = request.files['image']
        print(f"Processing image: {image_file.filename}")
        
        # Save the uploaded image temporarily to get its path
        temp_path = os.path.join('temp', image_file.filename)
        os.makedirs('temp', exist_ok=True)
        image_file.save(temp_path)
        
        # Generate ID from image path (same as preprocessing)
        search_image_id = hashlib.md5(temp_path.encode()).hexdigest()
        print(f"Generated search image ID: {search_image_id}")
        
        try:
            print("Opening image file...")
            image = Image.open(temp_path).convert('RGB')
            print("Image opened successfully")
            
            # Clean up temp file
            os.remove(temp_path)
            
        except Exception as e:
            print(f"ERROR opening image: {str(e)}")
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Extract features
        try:
            print("Extracting features from image...")
            query_features = extract_features(image)
            print("Features extracted successfully")
            print(f"Feature shape: {query_features.shape}")
            
            # Print first few values of query features
            print("Query feature sample:", query_features[:5])
            
        except Exception as e:
            print(f"ERROR extracting features: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        
        # Search similar items
        try:
            print("Computing similarities...")
            k = min(50, len(embeddings))
            
            # Print first few values of stored embeddings
            print("Stored embeddings sample (first item):", embeddings[0][:5])
            
            D, I = compute_distances(query_features, k)
            print(f"Found {len(I)} matches")
            
        except Exception as e:
            print(f"ERROR during similarity search: {str(e)}")
            return jsonify({'error': f'Error during similarity search: {str(e)}'}), 500
        
        # Process results
        try:
            print("\nProcessing search results...")
            results = []
            filtered_count = 0
            
            for idx, distance in zip(I, D):
                product_id = str(product_ids[idx])  # Ensure product_id is string
                similarity = 1 - distance  # Convert distance back to similarity
                
                # Debug print
                print(f"\nResult {len(results) + 1}:")
                print(f"Index in embeddings: {idx}")
                print(f"Product ID: {product_id}")
                print(f"Similarity score: {similarity:.4f}")
                
                # Check if product_id exists in metadata
                if product_id not in metadata_dict:
                    print(f"WARNING: Product ID {product_id} not found in metadata!")
                    continue
                    
                item = metadata_dict[product_id]
                print(f"Found in metadata: {item['category']}")
                
                # If similarity is very high, this might be the exact image
                if similarity > 0.99:
                    print("EXACT MATCH FOUND!")
                    print(f"Image URL: {item['image_url']}")
                
                # Apply filters
                if gender != 'all' and item['gender'].lower() != gender.lower():
                    print(f"Filtered out: wrong gender ({item['gender']})")
                    filtered_count += 1
                    continue
                    
                if category != 'all' and item['category'].lower() != category.lower():
                    print(f"Filtered out: wrong category ({item['category']})")
                    filtered_count += 1
                    continue
                
                # Create result item
                result_item = {
                    'id': product_id,
                    'score': float(distance),
                    'similarity': float(similarity),
                    'gender': item['gender'],
                    'category': item['category'],
                    'image_url': item['image_url'],
                    'name': item['name'],
                    'price': item['price'],
                    'details': item['details'],
                    'link': item['link']
                }
                
                results.append(result_item)
                
                if len(results) >= 20:
                    break
            
            print(f"\nReturning {len(results)} results (filtered out {filtered_count} items)")
            print("First result example:")
            print(json.dumps(results[0], indent=2))
            return jsonify(results)
            
        except Exception as e:
            print(f"ERROR processing results: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return jsonify({'error': f'Error processing results: {str(e)}'}), 500
            
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/search/keyword', methods=['POST'])
def search_by_keyword():
    try:
        data = request.get_json()
        keyword = data.get('keyword', '').lower()
        gender = data.get('gender', 'all')
        category = data.get('category', 'all')
        
        if not keyword:
            return jsonify({'error': 'No keyword provided'}), 400
        
        # Search in metadata
        results = []
        for product in metadata_dict.values():
            # Apply filters
            if gender != 'all' and product['gender'].lower() != gender.lower():
                continue
                
            if category != 'all' and product['category'].lower() != category.lower():
                continue
            
            # Search in all text fields
            searchable_text = ' '.join([
                product['name'].lower(),
                product['details'].lower(),
                product['category'].lower()
            ])
            
            if keyword in searchable_text:
                results.append(product)
            
            if len(results) >= 20:  # Limit results
                break
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in keyword search: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    try:
        gender = request.args.get('gender', 'all')
        
        # Get unique categories
        categories = set()
        for item in metadata_dict.values():
            if gender == 'all' or item['gender'].lower() == gender.lower():
                categories.add(item['category'])
        
        return jsonify(sorted(list(categories)))
    
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def initialize_app():
    print("\n=== Starting Application Initialization ===")
    
    # Initialize model
    print("\nInitializing model...")
    if not load_model():
        print("ERROR: Failed to load model!")
        return False
    print(" Model loaded successfully")
    
    # Initialize data
    print("\nLoading data...")
    if not load_data():
        print("ERROR: Failed to load data!")
        return False
    print(" Data loaded successfully")
    print(f" Loaded {len(metadata_dict)} items")
    print(f" Loaded {len(embeddings)} embeddings")
    
    print("\n=== Initialization Complete ===")
    return True

if __name__ == '__main__':
    try:
        print("\n=== Starting Flask Application ===")
        
        # Initialize the application
        if not initialize_app():
            print("ERROR: Failed to initialize application. Exiting...")
            exit(1)
            
        # Start the server
        print("\nStarting server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        exit(1)
