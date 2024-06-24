# Bible Embeddings (BGE-Large)

This repository contains pre-computed embeddings for the World English Bible (WEB) text, generated using the BAAI/bge-large-en-v1.5 model. These embeddings are designed for use in semantic search applications and other natural language processing tasks related to biblical text.

## Dataset Information

- **Source Text**: World English Bible (WEB)
- **Embedding Model**: BAAI/bge-large-en-v1.5
- **Embedding Dimensions**: 1024
- **Format**: JSON

## File Structure

The embeddings are stored in a single JSON file:

```
bible_embeddings.json
```

Each entry in the JSON file has the following structure:

```json
{
  "content": "Verse text",
  "metadata": {
    "book": "Book abbreviation",
    "chapter": chapter_number,
    "testament": "OT or NT"
  },
  "embedding": [float_array_of_1024_dimensions]
}
```

## Usage

These embeddings can be used for various NLP tasks, particularly semantic search applications. Here's a basic example of how to use them in a Python script:

```python
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the embeddings
with open('bible_embeddings.json', 'r') as f:
    bible_data = json.load(f)

# Function to find the most similar verses
def find_similar_verses(query_embedding, top_n=5):
    similarities = []
    for verse in bible_data:
        similarity = cosine_similarity(
            [query_embedding],
            [verse['embedding']]
        )[0][0]
        similarities.append((verse, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Example usage (assuming you have a function to generate query embeddings)
query_embedding = generate_query_embedding("river of life")
similar_verses = find_similar_verses(query_embedding)

for verse, similarity in similar_verses:
    print(f"{verse['metadata']['book']} {verse['metadata']['chapter']}:")
    print(f"{verse['content']}")
    print(f"Similarity: {similarity:.4f}\n")
```

## License

This dataset is derived from the World English Bible, which is in the public domain. The embeddings are provided under the [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).

## Acknowledgements

- The World English Bible (WEB) for the source text
- BAAI for the bge-large-en-v1.5 model

## Contact

For questions or issues regarding this dataset, please open an issue in this GitHub repository.
