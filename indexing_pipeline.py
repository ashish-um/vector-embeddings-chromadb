# indexing_pipeline.py

import os
import asyncio
from typing import List, Dict, Any

# Third-party libraries
from dotenv import load_dotenv
import google.generativeai as genai
import google.ai.generativelanguage as glm
import chromadb

# Import the function from your 1B solution
from document_parser import parse_document_to_sections

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "models/text-embedding-004"
CHROMA_COLLECTION_NAME = "hackathon_collection"


class IndexingPipeline:
    """
    A pipeline that uses a local ChromaDB instance for fast, consistent indexing.
    Now includes asynchronous methods for embedding and processing.
    """

    def __init__(self, google_api_key: str):
        if not google_api_key:
            raise ValueError("Google API key is required. Please check your .env file.")

        genai.configure(api_key=google_api_key)
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        print("IndexingPipeline for local ChromaDB initialized successfully.")

    def _prepare_chunk_for_embedding(self, chunk: Dict[str, Any]) -> str:
        full_path_str = " > ".join(chunk.get("full_path", []))
        return f"Section Path: {full_path_str}\nContent: {chunk.get('content', '')}"

    async def process_and_index_pdf_async(self, pdf_path: str, model_path: str, encoder_path: str):
        """
        Asynchronous version of the main orchestration method for processing a single PDF.
        """
        print("-" * 80)
        print(f"Starting async processing for: {os.path.basename(pdf_path)}")

        # 1. Parse Document (This remains synchronous as it's CPU-bound)
        print(f"Step 1: Parsing {os.path.basename(pdf_path)}...")
        parsed_data = parse_document_to_sections(pdf_path, model_path, encoder_path)
        if not parsed_data:
            print(f"Warning: No sections were parsed from {os.path.basename(pdf_path)}. Aborting.")
            return
        print(f"Successfully parsed into {len(parsed_data)} sections.")

        # 2. Generate Embeddings Asynchronously
        print(f"Step 2: Generating embeddings for {os.path.basename(pdf_path)}...")
        texts_to_embed = [self._prepare_chunk_for_embedding(chunk) for chunk in parsed_data]
        
        # Make the async call using keyword arguments, similar to the sync version
        result = await genai.embed_content_async(
            model=EMBEDDING_MODEL,
            content=texts_to_embed,
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        # The async response is a dictionary, just like the sync version
        embeddings = result['embedding']
        print(f"Successfully generated {len(embeddings)} embeddings for {os.path.basename(pdf_path)}.")

        # 3. Prepare and Upsert data (This is fast and local, so no async needed here)
        print(f"Step 3: Upserting {len(parsed_data)} documents to local ChromaDB...")
        documents_to_upsert = texts_to_embed
        metadatas_to_upsert = []
        ids_to_upsert = []

        for i, chunk in enumerate(parsed_data):
            metadatas_to_upsert.append({
                "document_name": chunk.get("document_name", ""),
                "page_number": int(chunk.get("page_number", 0)),
                "section_title": chunk.get("section_title", ""),
                "full_path": " > ".join(chunk.get("full_path", [])),
                "original_content": chunk.get("content", "")
            })
            ids_to_upsert.append(f"{os.path.basename(pdf_path)}_{i}")

        self.collection.add(
            embeddings=embeddings,
            documents=documents_to_upsert,
            metadatas=metadatas_to_upsert,
            ids=ids_to_upsert
        )
        
        print(f"Upsert complete for {os.path.basename(pdf_path)}.")
        print("-" * 80)
