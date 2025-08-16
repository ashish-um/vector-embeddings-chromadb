# app.py

import os
from flask import Flask, request, jsonify
from retrieval_handler import RetrievalHandler

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Initialize the RetrievalHandler ---
# This is done once when the server starts to avoid re-initializing on every request.
try:
    retrieval_handler = RetrievalHandler(google_api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"FATAL: Could not initialize RetrievalHandler: {e}")
    retrieval_handler = None

# --- API Endpoint ---
@app.route('/get_insights', methods=['POST'])
def get_insights():
    """
    API endpoint to get insights for a given text selection.
    Expects a JSON payload with a "selection" key.
    e.g., {"selection": "this is the text the user highlighted"}
    """
    if not retrieval_handler:
        return jsonify({"error": "Backend handler is not initialized."}), 500

    data = request.get_json()
    if not data or 'selection' not in data:
        return jsonify({"error": "Missing 'selection' key in request body."}), 400

    user_selection = data['selection']
    
    print(f"\nReceived request for selection: '{user_selection[:50]}...'")
    
    try:
        results = retrieval_handler.get_insights_for_selection(user_selection)
        return jsonify(results)
    except Exception as e:
        print(f"An error occurred during insight generation: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # The server will run on port 8080 as required by the hackathon rules.
    app.run(host='0.0.0.0', port=8080, debug=True)

