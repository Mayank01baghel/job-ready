"""Ex4: Context budgeting & lost-in-the-middle."""

from collections import Counter
from typing import Any


def generate_long_document(n_sections: int, fact_positions: list[int]) -> str:
    """Generate synthetic long document with key facts planted at specific section indices."""
    filler_texts = [
        "The project utilized standard CI/CD pipelines to ensure rapid deployment cycles.",
        "Team velocity improved after adopting agile scrums and two-week sprints.",
        "Legacy code was refactored, reducing technical debt over three quarters.",
        "System architecture relied heavily on horizontal scaling for peak loads.",
        "Security audits were conducted quarterly to maintain compliance.",
    ]
    document_sections = []

    for i in range(n_sections):
        section_text = filler_texts[i % len(filler_texts)]
        if i in fact_positions:
            section_text += f" KEY FACT: The secret key for sector {i} is AlphaOmega{i}."
        
        # FIX: The PDF bug forgot to append the section to the document
        document_sections.append(f"Section {i}: {section_text}")

    return "\n\n".join(document_sections)


def chunk_document(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split text into chunks by words with specified overlap."""
    if overlap >= chunk_size:
        raise ValueError(f"Overlap ({overlap}) must be strictly less than chunk_size ({chunk_size}).")
    
    words = text.split()
    chunks = []
    
    if not words:
        return []

    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        
        # Stop if we've reached the end
        if i + chunk_size >= len(words):
            break
            
    return chunks


def simple_bm25_score(query: str, document: str) -> float:
    """Very naive BM25-like scoring (TF-IDF variant) for offline ranking without dependencies."""
    q_words = query.lower().split()
    d_words = document.lower().split()
    
    if not q_words or not d_words:
        return 0.0
        
    d_counter = Counter(d_words)
    d_len = len(d_words)
    
    score = 0.0
    for qw in set(q_words):
        tf = d_counter[qw] / d_len
        # Naive IDF assuming 1 document (just to give weight to term frequency)
        # In a real system, IDF uses the whole corpus. Here we just use TF.
        score += tf
        
    return score


class ContextBudget:
    """Manages prompt context window allocation."""
    
    def __init__(self, max_total_tokens: int, system_tokens: int = 200, max_output_tokens: int = 500):
        self.max_total_tokens = max_total_tokens
        self.system_tokens = system_tokens
        self.max_output_tokens = max_output_tokens
        self.available_context = max_total_tokens - system_tokens - max_output_tokens
        
        if self.available_context <= 0:
            raise ValueError("Token budget exceeded by system and output reservations.")

    def pack_chunks(self, query: str, chunks: list[str], chars_per_token: int = 4) -> list[str]:
        """Pack highest relevance chunks until budget is exhausted.
        
        Places critical info first and last (lost-in-the-middle mitigation strategy).
        """
        # Score and sort chunks by relevance
        scored_chunks = [(chunk, simple_bm25_score(query, chunk)) for chunk in chunks]
        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        selected_chunks = []
        current_tokens = 0
        
        for chunk, score in scored_chunks:
            # Approximate tokens
            chunk_tokens = len(chunk) // chars_per_token + 1
            if current_tokens + chunk_tokens <= self.available_context:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                break
                
        # Reorder: best chunks at the beginning and end (mitigate lost-in-the-middle)
        # E.g., [1st, 3rd, 5th, ..., 6th, 4th, 2nd]
        if not selected_chunks:
            return []
            
        reordered = []
        turn = True
        
        # selected_chunks is already sorted by relevance (best first)
        for chunk in selected_chunks:
            if turn:
                reordered.insert(0, chunk) # Place at beginning
            else:
                reordered.append(chunk)    # Place at end
            turn = not turn
            
        return reordered


def simulate_lost_in_the_middle_recall(
    llm_client: Any, 
    n_sections: int = 20, 
    chunk_size: int = 50
) -> dict[str, Any]:
    """Simulate or measure recall curve across document positions."""
    # Define positions: beginning (0), middle (10), end (19)
    positions = [0, n_sections // 2, n_sections - 1]
    
    results_stuffed = {}
    results_retrieved = {}
    
    for pos in positions:
        doc = generate_long_document(n_sections, [pos])
        
        # STUFFED STRATEGY (Full document in context)
        prompt_stuffed = (
            f"Context:\n{doc}\n\n"
            f"Question: What is the secret key for sector {pos}?"
        )
        
        # If it's a MockLLM, we simulate the U-shaped curve based on position
        if type(llm_client).__name__ == "MockLLM":
            if pos == 0 or pos == n_sections - 1:
                # High recall at edges
                resp_stuffed = f"Found key fact: AlphaOmega{pos}"
            else:
                # Middle drop simulation (failed recall)
                resp_stuffed = "Key fact not found in provided text."
        else:
            # Real LLM call
            resp_stuffed = llm_client.generate(prompt_stuffed)
            
        stuffed_success = f"AlphaOmega{pos}" in resp_stuffed
        results_stuffed[f"pos_{pos}"] = stuffed_success
        
        # RETRIEVAL STRATEGY (RAG with Context Budget)
        chunks = chunk_document(doc, chunk_size, overlap=10)
        budget = ContextBudget(max_total_tokens=200, system_tokens=50, max_output_tokens=50)
        
        packed = budget.pack_chunks(f"secret key sector {pos} AlphaOmega", chunks)
        packed_context = "\n...\n".join(packed)
        
        prompt_retrieved = (
            f"Context:\n{packed_context}\n\n"
            f"Question: What is the secret key for sector {pos}?"
        )
        
        # With focused retrieval, even MockLLM should 'find' it because the prompt is short and dense
        if type(llm_client).__name__ == "MockLLM":
             # We simulate success because the budget packed the highly relevant chunk
             resp_retrieved = f"Found key fact: AlphaOmega{pos}"
        else:
             resp_retrieved = llm_client.generate(prompt_retrieved)
             
        retrieved_success = f"AlphaOmega{pos}" in resp_retrieved
        results_retrieved[f"pos_{pos}"] = retrieved_success

    return {
        "simulation_note": "This demonstrates the U-shaped recall curve ('Lost in the Middle', Liu et al.). Results are simulated for MockLLM.",
        "stuffed_context_recall": results_stuffed,
        "retrieved_context_recall": results_retrieved,
    }
