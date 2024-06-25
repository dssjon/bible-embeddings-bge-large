# Bible Semantic Search App

This project is a Bible exploration tool that allows users to perform semantic searches over the entire Bible text using state-of-the-art NLP techniques. It has evolved from a server-side Python application to a client-side JavaScript app, enabling free hosting and improved accessibility.

## Features

- Semantic search over the entire Bible text
- Client-side vector similarity search using Transformers.js
- Web UI built with HTML, Tailwind CSS, and vanilla JavaScript
- Supports filtering by Old Testament and New Testament
- Displays top search results with similarity scores

## Architecture

The app follows a client-side architecture:

1. Bible text is preprocessed and stored as JSON files, including pre-computed embeddings
2. The web app loads these JSON files and the Transformers.js model in the browser
3. User searches for a topic, and relevant passages are retrieved by semantic similarity computed entirely in the browser
4. Top results are displayed to the user with similarity scores

This approach enables combining the strengths of dense vector search for retrieval with the accessibility and cost-effectiveness of a static web application.

## Running the App Locally

To run the Bible Search App locally:

1. Clone this repository:

2. Serve the directory with a local web server. For example, using Python's built-in HTTP server:
   ```
   python -m http.server 8000
   ```

3. Open a web browser and navigate to `http://localhost:8000`

## Data Preprocessing

The Bible text and embeddings are preprocessed and stored in JSON files. If you need to regenerate this data:

1. Ensure you have Python 3.7+ installed
2. Install required libraries:
   ```
   pip install langchain transformers torch
   ```
3. Run the embedding generation script (adjust paths as needed):
   ```
   python generate_embeddings.py -i "./engwebp_vpl.xml" -m "BAAI/bge-large-en-v1.5" -o "./public"
   ```

## Technologies Used

- [Transformers.js](https://github.com/xenova/transformers.js) - Running transformer models in the browser
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) - Text embedding model
- [Vercel](https://vercel.com/) - Hosting and deployment platform

## Credits

This project builds upon the work of the following open-source projects and resources:

- [World English Bible (WEB)](https://worldenglish.bible/) - Source text
- [Langchain](https://github.com/langchain-ai/langchain) - Used in preprocessing for building LLMs through composability
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - State-of-the-art Natural Language Processing

## License

This project is open-source and available under the MIT License.

## Contact

For questions or issues regarding this project, please open an issue in this GitHub repository.