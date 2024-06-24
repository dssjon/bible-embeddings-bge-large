# Bible Embeddings Generator

This project generates embeddings for the World English Bible (WEB) text using the BAAI/bge-large-en-v1.5 model. These embeddings are designed for use in semantic search applications and other natural language processing tasks related to biblical text.

## Features

- Generates embeddings for each chapter of the Bible
- Uses the BAAI/bge-large-en-v1.5 model for high-quality embeddings
- Supports customization of input file, model, and output directory
- Saves embeddings in JSON format, organized by book

## Requirements

- Python 3.7+
- Required libraries: langchain, transformers, torch, xml.etree.ElementTree, argparse

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bible-embeddings-generator.git
   cd bible-embeddings-generator
   ```

2. Install the required libraries:
   ```
   pip install langchain transformers torch
   ```

## Usage

Run the script with the following command:

```
python generate_embeddings.py [-i INPUT_FILE] [-m MODEL_NAME] [-q QUERY_INSTRUCTION] [-o OUTPUT_DIR]
```

Arguments:
- `-i`, `--input_file`: Path to input VPL file (default: "./engwebp_vpl.xml")
- `-m`, `--model_name`: Name of the HuggingFace model to use (default: "BAAI/bge-large-en-v1.5")
- `-q`, `--query_instruction`: Query instruction for embedding generation (default: "Represent the Religious Bible verse text for semantic search:")
- `-o`, `--output_dir`: Path to base output directory (default: "./output_bible_db")

Example:
```
python generate_embeddings.py -i "./my_bible.xml" -m "BAAI/bge-base-en" -o "./my_embeddings"
```

## Output

The script generates JSON files for each book of the Bible, stored in the specified output directory. Each JSON file contains an array of objects with the following structure:

```json
{
  "content": "Chapter text",
  "metadata": {
    "book": "Book name",
    "chapter": chapter_number,
    "testament": "OT or NT"
  },
  "embedding": [float_array_of_1024_dimensions]
}
```

## Using the Embeddings

These embeddings can be used for various NLP tasks, particularly semantic search applications. Here's a basic example of how to use them in a Python script:

```python
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the embeddings
def load_embeddings(book_name):
    with open(f'./output_bible_db/{book_name.lower()}.json', 'r') as f:
        return json.load(f)

# Function to find the most similar verses
def find_similar_verses(query_embedding, book_data, top_n=5):
    similarities = []
    for chapter in book_data:
        similarity = cosine_similarity(
            [query_embedding],
            [chapter['embedding']]
        )[0][0]
        similarities.append((chapter, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Example usage (assuming you have a function to generate query embeddings)
query_embedding = generate_query_embedding("river of life")
book_data = load_embeddings("rev")
similar_verses = find_similar_verses(query_embedding, book_data)

for chapter, similarity in similar_verses:
    print(f"{chapter['metadata']['book']} {chapter['metadata']['chapter']}:")
    print(f"{chapter['content'][:100]}...")  # Print first 100 characters
    print(f"Similarity: {similarity:.4f}\n")
```

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- The World English Bible (WEB) for the source text
- BAAI for the bge-large-en-v1.5 model

## Contact

For questions or issues regarding this project, please open an issue in this GitHub repository.
