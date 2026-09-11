"""
Advanced LLM service with query rewriting, contextual compression,
and active fallback logic for production RAG.

Features:
- Query rewriting to improve retrieval
- Contextual compression to reduce token usage
- Temperature lockdown (0.0) for deterministic responses
- JSON output mode for structured results
- Confidence-based fallback when retrieval quality is low
"""
import os
import json
from duckduckgo_search import DDGS
from .compressor import ContextualCompressor


class LLMService:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", confidence_threshold: float = 0.7):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold  # Fallback if score < threshold
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = None
        self.compressor = ContextualCompressor()
        
        if self.groq_api_key:
            try:
                from groq import Groq

                self.groq_client = Groq(api_key=self.groq_api_key)
            except ImportError:
                self.groq_client = None
            except Exception:
                self.groq_client = None

    def _rewrite_query(self, query: str) -> str:
        """Query rewriting to improve retrieval.
        
        This expands the query or rephrases it to be more specific.
        In production, this would use an LLM to generate multiple query variants.
        
        For now, we use simple heuristics:
        - Expand acronyms
        - Add related keywords
        """
        expanded = query
        
        # Simple acronym expansion (example)
        acronym_map = {
            "bms": "Bristol Myers Squibb",
            "fda": "Food and Drug Administration",
            "rag": "Retrieval Augmented Generation",
        }
        
        for acronym, expansion in acronym_map.items():
            if acronym.lower() in query.lower():
                expanded += f" {expansion}"
        
        return expanded

    def _search_web(self, query: str, max_results: int = 3) -> list:
        """Web search fallback using DuckDuckGo."""
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
            if not results:
                return []
            return [
                {
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "href": item.get("href", ""),
                }
                for item in results
            ]
        except Exception:
            return []

    def _format_web_context(self, results: list) -> str:
        """Format web search results for inclusion in prompt."""
        return "\n---\n".join(
            [
                f"Web source {i+1}: {item['title']} - {item['body']} ({item['href']})"
                for i, item in enumerate(results)
            ]
        )

    def _compress_context(self, query: str, texts: list) -> str:
        """Compress retrieved text using contextual compression.
        
        This extracts only the most relevant sentences from each passage,
        reducing token usage and improving LLM focus.
        """
        compressed_parts = []
        for text_dict in texts:
            text = text_dict.get("text", "") if isinstance(text_dict, dict) else text_dict
            compressed = self.compressor.compress(query, text)
            if compressed:
                compressed_parts.append(compressed)
        
        return "\n---\n".join(compressed_parts)

    def _check_retrieval_quality(self, retrieved_texts: list) -> float:
        """Check if retrieval quality meets the confidence threshold.
        
        Returns the average confidence score.
        If below the threshold, we should use fallback retrieval.
        """
        if not retrieved_texts:
            return 0.0
        
        scores = [t.get("score", 0.0) for t in retrieved_texts]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score

    def generate_answer(self, query: str, retrieved_texts: list):
        """Generate an answer with query rewriting, compression, and fallback.
        
        This is the complete optimized generation pipeline:
        1. Rewrite the query to improve it
        2. Check retrieval quality
        3. Compress context if retrieval is good
        4. Generate answer with Groq or fallback
        5. Enforce deterministic output (temperature=0.0)
        6. Use JSON mode for structured output
        """
        # Step 1: Query rewriting
        rewritten_query = self._rewrite_query(query)
        
        # Step 2: Check retrieval quality (confidence threshold)
        retrieval_quality = self._check_retrieval_quality(retrieved_texts)
        
        # Step 3: Active Fallback Logic
        use_web_search = retrieval_quality < self.confidence_threshold
        
        # Step 4: Prepare context
        if retrieved_texts and retrieval_quality >= self.confidence_threshold:
            # Good retrieval: compress and use
            source_context = self._compress_context(rewritten_query, retrieved_texts)
            context_source = "indexed_pdf"
        else:
            source_context = ""
            context_source = "web_search"
        
        # Step 5: Web search fallback if needed
        web_results = []
        if use_web_search or not source_context:
            web_results = self._search_web(rewritten_query)
        
        web_context = self._format_web_context(web_results) if web_results else ""
        
        # Step 6: Build the system prompt (STRICT CONTRACT)
        system_prompt = (
            "You are a professional pharmaceutical market analyst. "
            "Respond ONLY with JSON in this format: "
            '{"answer": "your answer here", "confidence": "high|medium|low", "source": "pdf|web"} '
            "Keep responses concise and professional. "
            "Do not include markdown, extra text, or formatting outside the JSON."
        )
        
        # Step 7: Build user instruction
        prompt_parts = [f"Question: {rewritten_query}"]
        
        if source_context:
            prompt_parts.append(f"PDF Sources:\n{source_context}")
        
        if web_context:
            prompt_parts.append(f"Web Results:\n{web_context}")
        
        if not source_context and not web_context:
            prompt_parts.append(
                "No sources found. Please provide a helpful response based on your training data."
            )
        
        prompt_parts.append("Respond in JSON format only.")
        user_instruction = "\n\n".join(prompt_parts)
        
        # Step 8: Call Groq with locked parameters
        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_instruction},
                    ],
                    temperature=0.0,  # LOCKED: Deterministic responses
                    max_tokens=512,
                    response_format={"type": "json_object"},  # JSON mode
                )
                response_text = completion.choices[0].message.content.strip()
                
                # Parse JSON response
                try:
                    result_json = json.loads(response_text)
                    answer = result_json.get("answer", "No answer generated.")
                    confidence = result_json.get("confidence", "unknown")
                    source = result_json.get("source", context_source)
                    return f"[{confidence.upper()}] {answer} (Source: {source})"
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    return f"[MEDIUM] {response_text}"
            except Exception as e:
                # Groq call failed: use fallback
                return self._fallback_answer(query, source_context, web_context)
        
        # No Groq: use fallback logic
        return self._fallback_answer(query, source_context, web_context)

    def _fallback_answer(self, query: str, source_context: str, web_context: str) -> str:
        """Generate a fallback answer when Groq is not available.
        
        This ensures we always provide a response, even with low confidence.
        """
        if source_context:
            # Use best passage from PDF
            first_line = source_context.split("\n")[0][:300]
            return f"[LOW] Based on indexed PDF: {first_line}"
        elif web_context:
            # Use web search result
            first_line = web_context.split("\n")[0][:300]
            return f"[LOW] Based on web search: {first_line}"
        else:
            # Complete fallback
            return f"[CRITICAL] Unable to find relevant information about: {query}"
