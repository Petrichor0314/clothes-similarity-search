# Clothes Similarity Search

This project implements a clothing similarity search system. The system uses deep learning to find similar clothing items based on image features.

## Project Structure
- `app.py`: Main Flask application that serves the API endpoints
- `preprocess.py`: Script for preprocessing images and generating embeddings
- `templates/`: Contains the HTML templates
- `dataset/`: Directory containing the clothing images
- `embeddings.json`: Pre-computed embeddings for the clothing items
- `metadata.json`: Metadata information for the clothing items

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## API Endpoints

- `POST /search`: Accepts an image file and returns similar clothing items
- `GET /random_images`: Returns a set of random images from the dataset

## Integration Points for E-commerce

To integrate this similarity search system with an e-commerce frontend:

1. The search endpoint (`/search`) can be called with a POST request containing an image file
2. The response includes similar items with their metadata, which can be linked to your product database
3. The system can be extended by adding product IDs and prices to the metadata

## Requirements

- Python 3.8+
- CUDA-compatible GPU recommended for faster processing
- See requirements.txt for all Python dependencies

## Notes
- The embeddings.json file contains pre-computed embeddings - keep this file in the project root
- Make sure to maintain the same directory structure when deploying
