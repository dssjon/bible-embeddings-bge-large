import json
import os
import xml.etree.ElementTree as ET
import collections
import sys
import argparse
from datetime import datetime

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings

# Accept the following arguments:
#  -input_file (-i) : path to input VPL file (default: "./engwebp_vpl.xml")
#  -model_name (-m) : name of the HuggingFace model to use (default: "BAAI/bge-large-en-v1.5")
#  -query_instruction (-q) : query instruction to use (default: "Represent the religious Bible verse text for semantic search:")
#  -output_dir (-o) : path to base output directory (default: "./output_bible_db")

input_file = "./engwebp_vpl.xml"
model_name = "BAAI/bge-large-en-v1.5"
query_instruction = "Represent the Religious Bible verse text for semantic search:"
output_dir = "./output_bible_db"

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", default=input_file, help=f"path to input VPL file (expected .xml format, default {input_file})")
parser.add_argument("-m", "--model_name", default=model_name, help=f"name of the HuggingFace model to use (default: {model_name})")
parser.add_argument("-q", "--query_instruction", default=query_instruction, help=f"query instruction to use (default: \"{query_instruction}\")")
parser.add_argument("-o", "--output_dir", default=output_dir, help="path to base output directory. The output directory will be modified to reflect the input_file and model_name parameters if they are different from their defaults.")
args = parser.parse_args()

output_dir = args.output_dir

# If any of the arguments are not at their default, then modify the output_dir to reflect the arguments
if args.input_file != input_file:
    input_file = args.input_file
    output_dir = output_dir + "_" + input_file.split('/')[-1].split('.')[0]
if args.model_name != model_name:
    model_name = args.model_name
    output_dir = output_dir + "_" + model_name.replace("/", "_")
if args.query_instruction != query_instruction:
    query_instruction = args.query_instruction

print(f"input_file: {input_file}")
print(f"model_name: {model_name}")
print(f"query_instruction: {query_instruction}")
print(f"output_dir: {output_dir}")

# Load XML
tree = ET.parse(input_file)
root = tree.getroot()

then = datetime.now()

print()
print("Parsing XML, grouping verses by book and chapter...")
# Group verses by book and chapter
verses_by_book_chapter = collections.defaultdict(lambda: collections.defaultdict(list))
for verse in root.findall("v"):
    book = verse.attrib["b"]
    chapter = int(verse.attrib["c"])
    verse_num = int(verse.attrib["v"])
    text = verse.text

    verses_by_book_chapter[book][chapter].append((verse_num, text))

print(f' {sum(sum(len(verses) for verses in chapters.values()) for chapters in verses_by_book_chapter.values())} verses found')
print(f' {len(verses_by_book_chapter)} books found')

print(f'Creating documents for each chapter...')
# Create document for each chapter
documents_by_book = collections.defaultdict(list)
testament = "OT"

for book, chapters in verses_by_book_chapter.items():
    for chapter, verses in chapters.items():
        chapter_text = ""
        for verse_num, text in verses:
            chapter_text += f"{text}\n"

        if book.lower().startswith("mat"):
            testament = "NT"

        doc = Document(page_content=chapter_text)
        doc.metadata = {
            "book": book,
            "chapter": chapter,
            "testament": testament,
        }
        documents_by_book[book].append(doc)

# Split into chunks
chunk_size = 1500
chunk_overlap = 0
verse_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# Load embeddings
print(f"Loading embeddings from model {model_name}...")
embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    query_instruction=query_instruction,
    encode_kwargs={'normalize_embeddings': True},
    model_kwargs={"device": "mps"}
)

print("Generating embeddings and saving to JSON files by book...")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate embeddings and save to JSON files by book
for book, documents in documents_by_book.items():
    print(f"Processing book: {book}")
    
    # Split documents for this book
    bible_book = verse_splitter.split_documents(documents)
    
    embeddings_data = []
    for i, doc in enumerate(bible_book):
        # Generate embedding
        embedding = embedding_function.embed_query(doc.page_content)
        
        # Create a dictionary with document info and embedding
        doc_data = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "embedding": embedding
        }
        
        embeddings_data.append(doc_data)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} documents for {book}")

    # Save to JSON file for this book
    output_file = os.path.join(output_dir, f"{book.lower()}.json")
    with open(output_file, 'w') as f:
        json.dump(embeddings_data, f)
    
    print(f"Embeddings for {book} saved to {output_file}")

print("Done!")

completed_at = datetime.now()
elapsed_time_s = (completed_at - then).total_seconds()

print(f"Completed in {elapsed_time_s} seconds")
