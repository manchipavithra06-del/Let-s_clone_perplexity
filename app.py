import os
import io
import math
from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
from google import genai
import PyPDF2
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize GenAI client
# Using user's api key
client = genai.Client(api_key="AIzaSyAHBSwqdoEgyfkTlnC5zuG8xMDtFeUzS9o")

# In-memory storage for PDF chunks and their embeddings
# For a production app, use a vector database like ChromaDB or Pinecone
document_store = []

def embed_text(text):
    """Generate an embedding for the given text using gemini-embedding-001."""
    try:
        response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=text
        )
        # Depending on SDK return type, extract the array
        values = response.embeddings[0].values
        return np.array(values)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def compute_cosine_similarity(vec1, vec2):
    """Basic cosine similarity between two numpy arrays."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/agent_logo.png')
def logo():
    return send_file('agent_logo.png')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '')
    uploaded_file = request.files.get('file')

    if not user_message and not uploaded_file:
        return jsonify({'error': 'No message or file provided'}), 400

    # Process PDF if included
    if uploaded_file and uploaded_file.filename.endswith('.pdf'):
        try:
            # Read PDF from memory
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            full_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            # Simple Chunking: Split by paragraphs or roughly 1000 characters
            # Here we split by newlines, group them until ~1000 chars
            paragraphs = full_text.split('\n')
            current_chunk = ""
            chunks = []
            
            for p in paragraphs:
                if len(current_chunk) + len(p) < 1000:
                    current_chunk += p + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = p + " "
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Generate embeddings and store them
            global document_store
            document_store = [] # Clear old doc
            print(f"Embedding {len(chunks)} chunks...")
            for chunk in chunks:
                if len(chunk) < 10:
                    continue  # skip tiny chunks
                vec = embed_text(chunk)
                if vec is not None:
                    document_store.append({
                        "text": chunk,
                        "embedding": vec
                    })
            print("Finished embedding document.")
        except Exception as e:
            return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500

    # Implement RAG if document is loaded
    context_text = ""
    if document_store and user_message:
        query_vec = embed_text(user_message)
        if query_vec is not None:
            # Rank chunks
            results = []
            for doc in document_store:
                similarity = compute_cosine_similarity(query_vec, doc['embedding'])
                results.append((similarity, doc['text']))
            
            # Sort by highest similarity
            results.sort(key=lambda x: x[0], reverse=True)
            
            # Take top 3 chunks as context
            top_chunks = [r[1] for r in results[:3]]
            context_text = "\n\n".join(top_chunks)

    # Prepare prompt
    final_prompt = user_message
    if context_text:
        final_prompt = (
            "You are a helpful AI assistant. Use the following context from the user's uploaded document "
            "to answer their question. If the answer is not in the context, just answer normally but let them know.\n\n"
            f"--- CONTEXT ---\n{context_text}\n--- END CONTEXT ---\n\n"
            f"User Question: {user_message}"
        )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=final_prompt,
        )
        return jsonify({'reply': response.text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
