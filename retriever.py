from pinecone import Pinecone
import os
from dotenv import load_dotenv
from schema import RetrievedChunk # Import your schema
# Load environment variables from .env file
load_dotenv()

# Now this will work
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("dronacharya")
EMBED_MODEL = "llama-text-embed-v2"

def get_relevant_context(query: str, top_k: int = 3):
    # Generate embedding for the query
    res = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query"}
    )
    query_vec = res[0].values
    
    # Search Pinecone
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    
    # chunks = []
    # for match in results['matches']:
    #     chunks.append({
    #         'content': match['metadata']['text_content'],
    #         'relevance_reason': f"Similarity score: {round(match['score'], 4)}"
    #     })
    

    chunks = []
    for match in results['matches']:
        # Create the object instead of a dictionary
        chunks.append(RetrievedChunk(
            content=match['metadata']['text_content'],
            source_metadata="N/A", # Add this since your schema requires it
            relevance_reason=f"Similarity score: {round(match['score'], 4)}"
        ))
    return chunks