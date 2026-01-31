import json
import os
from pinecone import Pinecone, ServerlessSpec

# 1. Configuration - Add your API Key here
PINECONE_API_KEY = "..."
INDEX_NAME = "dronacharya"
EMBED_MODEL = "llama-text-embed-v2"  # As per your documentation request

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2. Create Index if it doesn't exist
# llama-text-embed-v2 typically outputs 1024 or 3072 dimensions. 
# We'll use 1024 for this example.
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

def process_and_upload(file_path):
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vectors_to_upsert = []

    for item in data:
        # Prepare the text for embedding
        # For Book chunks, we use title + content. 
        # For Notes, we use question + answer.
        text_to_embed = ""
        if "content" in item:
            text_to_embed = f"{item.get('title', '')} {item['content']}"
        else:
            text_to_embed = f"{item['question']} {item['answer']}"

        # 3. Generate Cloud Embedding using Pinecone Inference API (FREE TIER)
        # Note: 'input_type' can be 'passage' for storage
        embedding_response = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=[text_to_embed],
            parameters={"input_type": "passage", "truncate": "END"}
        )
        
        vector_values = embedding_response[0].values

        # 4. Prepare Metadata
        # We preserve every field from your metadata request
        metadata = item["metadata"]
        metadata["text_content"] = text_to_embed # Store text for retrieval
        
        vectors_to_upsert.append({
            "id": item["chunk_id"],
            "values": vector_values,
            "metadata": metadata
        })

    # 5. Upsert in batches of 50
    for i in range(0, len(vectors_to_upsert), 50):
        batch = vectors_to_upsert[i : i + 50]
        index.upsert(vectors=batch)
    
    print(f"Successfully uploaded {len(vectors_to_upsert)} chunks from {file_path}")

# Run for both files
try:
    process_and_upload("data/chapter1_2.json")
    process_and_upload("data/chapter1_notes.json")
    print("\nAll data is now stored in Pinecone.")
except FileNotFoundError as e:
    print(f"Error: Ensure your JSON files are in the same folder as this script. {e}")
except Exception as e:
    print(f"An error occurred: {e}")