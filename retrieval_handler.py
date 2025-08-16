# retrieval_handler.py

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-1.5-flash-latest" # Using the latest flash model for speed
CHROMA_COLLECTION_NAME = "hackathon_collection"

# This is the prompt template that instructs the LLM how to behave.
INSIGHTS_PROMPT_TEMPLATE = """
You are an expert AI assistant for a document analysis tool.
Your task is to analyze a user's selected text and, based ONLY on the provided context from their document library, generate three types of insights.
The context below consists of sections from documents the user has previously read.

Analyze the user's selection in light of the context and provide the following:
1.  **Contradictions (⚔️):** Identify any statements in the context that directly contradict or challenge the user's selected text. If none, state that.
2.  **Enhancements (💡):** Find any information in the context that builds upon, refines, or provides a more detailed example of the concept in the user's selection. If none, state that.
3.  **Connections (🔗):** Point out any related concepts or similar ideas from the context that are not direct enhancements but are relevant to the user's selection. If none, state that.

Your response must be concise, grounded in the provided context, and structured clearly under these three headings.

---
CONTEXT:
{context}
---

USER'S SELECTED TEXT:
"{user_selection}"

---
INSIGHTS:
"""


class RetrievalHandler:
    """
    Handles querying the vector database and generating insights using an LLM.
    """

    def __init__(self, google_api_key: str):
        if not google_api_key:
            raise ValueError("Google API key is required.")

        genai.configure(api_key=google_api_key)
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        self.generation_model = genai.GenerativeModel(GENERATION_MODEL)
        print("RetrievalHandler initialized successfully.")

    def retrieve_relevant_sections(self, user_selection: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds the user's query and retrieves the top_k most relevant sections
        from the ChromaDB collection.
        """
        print(f"Step 1: Retrieving top {top_k} relevant sections for selection...")
        
        # Embed the user's selection using the same model
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=user_selection,
            task_type="RETRIEVAL_QUERY" # Use RETRIEVAL_QUERY for search queries
        )
        query_embedding = result['embedding']

        # Query the collection
        query_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # The results contain documents, metadatas, distances, etc.
        # We are interested in the metadatas of the retrieved chunks.
        retrieved_metadatas = query_results.get('metadatas', [[]])[0]
        print(f"Found {len(retrieved_metadatas)} relevant sections.")
        return retrieved_metadatas

    def generate_insights(self, user_selection: str, context_sections: List[Dict[str, Any]]) -> str:
        """
        Generates categorized insights using the Gemini model based on the
        retrieved context.
        """
        if not context_sections:
            return "No relevant context was found in your library to generate insights."

        print("Step 2: Generating insights with Gemini...")
        
        # Format the context for the prompt
        context_str = "\n\n---\n\n".join(
            [f"From '{item.get('document_name', 'N/A')}' (Page {item.get('page_number', 'N/A')}):\n{item.get('original_content', '')}" 
             for item in context_sections]
        )

        # Create the full prompt
        prompt = INSIGHTS_PROMPT_TEMPLATE.format(
            context=context_str,
            user_selection=user_selection
        )
        
        try:
            response = self.generation_model.generate_content(prompt)
            print("Successfully received insights from Gemini.")
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "Error: Could not generate insights due to an API issue."

    def get_insights_for_selection(self, user_selection: str):
        """
        Main orchestration method to get both retrieved sections and generated insights.
        """
        retrieved_sections = self.retrieve_relevant_sections(user_selection)
        generated_insights = self.generate_insights(user_selection, retrieved_sections)
        
        return {
            "retrieved_sections": retrieved_sections,
            "generated_insights": generated_insights
        }

