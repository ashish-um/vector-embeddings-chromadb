# run_indexing_async.py

import os
import asyncio
import time
from indexing_pipeline import IndexingPipeline

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_PATH = "models/heading_classifier_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"
PDF_DIRECTORY = "pdfs" 


async def main():
    """
    Main asynchronous function to run the indexing pipeline concurrently for all PDFs.
    """
    start_time = time.time()

    # 1. Configuration Checks
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
        return
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print(f"Error: ML model files not found.")
        return
    if not os.path.isdir(PDF_DIRECTORY):
        print(f"Error: PDF directory not found at '{PDF_DIRECTORY}'.")
        return

    # 2. Initialize the pipeline (only once)
    try:
        pipeline = IndexingPipeline(google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    # 3. Create a list of tasks to run in parallel
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIRECTORY}'.")
        return

    print(f"\nFound {len(pdf_files)} PDFs. Starting concurrent processing...")
    
    tasks = []
    for pdf_file in pdf_files:
        full_path = os.path.join(PDF_DIRECTORY, pdf_file)
        # Create a coroutine for each PDF processing task
        task = pipeline.process_and_index_pdf_async(
            pdf_path=full_path,
            model_path=MODEL_PATH,
            encoder_path=ENCODER_PATH
        )
        tasks.append(task)

    # 4. Run all tasks concurrently and wait for them to complete
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"\nAll documents have been processed and indexed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    # Run the main asynchronous function
    asyncio.run(main())
