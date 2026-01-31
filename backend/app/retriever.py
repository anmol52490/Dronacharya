import os
from pinecone import Pinecone
from dotenv import load_dotenv
from .schema import RetrievedChunk

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("dronacharya")
EMBED_MODEL = "llama-text-embed-v2"

def get_relevant_context(query: str, top_k: int = 3):
    """Fetches relevant context from Pinecone."""
    try:
        res = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=[query],
            parameters={"input_type": "query"}
        )
        query_vec = res[0].values

        results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)

        chunks = []
        for match in results["matches"]:
            chunks.append(RetrievedChunk(
                content=match["metadata"].get("text_content", ""),
                source_metadata="Textbook",
                relevance_reason=f"Similarity: {round(match['score'], 4)}"
            ))
        return chunks
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return []
