import json
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Set
from dataclasses import dataclass
# Import messages for building the new prompt
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pathlib import Path  # <-- ADDED

# --- USER: EDIT THIS ---
#
# Assumptions:
# 1. Your class with the .rag() method is in a file named `storage.py`.
# 2. The class itself is named `LegislatureStorage`.
# 3. Your `Retrieval` object is also defined in `storage.py`.
# 4. Your `llm` and `prompt` are in `llm.py`.
#
# Please CHANGE these imports to match your actual project structure.
from load import Storage, Retrieval  # <-- CHANGED
from llm import llm                       # <-- CHANGED


# 1. Define the Pydantic Schema for a single article
class Article(BaseModel):
    """A well-structured article with a title, summary, and body."""
    article_title: str = Field(..., description="A catchy, descriptive title for the article.")
    article_summary: str = Field(..., description="A concise, one-paragraph summary of the article's main points.")
    article_body: str = Field(
        ..., 
        description="The full body of the article, written in plain text. Do not use markdown."
    )

# 2. Define the Pydantic Schema for the list of 5 articles
# This is what we'll ask the LLM to generate in one go.
class ArticleSet(BaseModel):
    """A collection of 5 distinct articles on a given topic."""
    articles: List[Article] = Field(
        ..., 
        description="A list of 5 unique articles covering different aspects of the topic."
    )


# 3. Define the topics from your image
TOPICS = [
    "Housing & Development",
    "Education Funding & Property Tax",
    "Taxes & Economic Policy",
    "Environment & Climate",
    "Workforce & Labor",
    "Healthcare & Mental Health",
    "Public Safety & Justice",
    "Infrastructure & Energy",
    "Civic & Electoral Reform"
]

# 4. REMOVED ASPECT_PROMPTS, as requested

# 5. Define the new System Prompt for the LLM
SYSTEM_PROMPT = """You are an expert legislative reporter and journalist. Your task is to write 5 distinct, high-quality articles based *only* on the provided legislative context.

Each of the 5 articles MUST cover a different aspect of the main topic. For example:
1.  A summary of key bills passed or debated.
2.  The financial implications and economic impact.
3.  The main arguments for and against proposals (key debates).
4.  Public testimony, community concerns, and stakeholder feedback.
5.  Future outlook, upcoming priorities, or next steps.

Do not just summarize the documents. Synthesize the information into 5 complete, well-written, journalistic articles. Ensure the articles are different from each other and provide unique value.
"""


def generate_all_articles():
    """
    Main function to loop through all topics and aspects,
    call the RAG function, and save the results to JSON.
    """
    
    # --- USER: EDIT THIS ---
    # Initialize your storage class here.
    # This instance must have the .rag() method.
    try:
        BASE_DIR = Path(__file__).resolve().parent
        FAISS_PATH = str((BASE_DIR.parent / "faiss_index").resolve())
        storage = Storage(path=FAISS_PATH, from_path=True)
    except Exception as e:
        print(f"!! ERROR: Failed to initialize your `Storage` class: {e}")  # <-- CHANGED
        print("Please make sure the class is imported correctly and can be initialized.")
        return
    # --- END USER EDIT ---

    all_articles_data = []
    total_topics = len(TOPICS)
    article_count = 0

    print(f"Starting article generation for {total_topics} topics...")

    for i, topic in enumerate(TOPICS):
        print(f"\n--- Processing Topic {i+1}/{total_topics}: {topic} ---")
        
        try:
            # 1. Make one RAG query per category, with no schema
            print(f"  (Step 1/3) Retrieving documents for '{topic}'...")
            rag_question = f"All relevant legislative documents (transcripts, bills, journals) regarding {topic}"
            
            retrieval: Retrieval = storage.rag(
                question=rag_question,
                schema=None,  # <-- Set to None as requested
                date_range=None
            )
            
            # 2. Collect document content and URLs
            documents: List[Any] = retrieval['documents']
            if not documents:
                print(f"  !! WARNING: No documents found for topic '{topic}'. Skipping.")
                continue

            docs_content = "\n\n".join(doc.page_content for doc in documents)
            
            referenced_urls: Set[str] = set()
            for doc in documents:
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) and doc.metadata.get('source_url'):
                    referenced_urls.add(doc.metadata['source_url'])
            
            print(f"  (Step 2/3) Retrieved {len(documents)} documents. Generating 5 articles...")

            # 3. Build the prompt for the LLM to generate 5 articles
            human_prompt = f"""Here is the legislative context on the topic of "{topic}":
            
            --- BEGIN CONTEXT ---
            {docs_content}
            --- END CONTEXT ---
            
            Please generate 5 complete, distinct articles based *only* on this context,
            following the schema provided.
            """
            
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=human_prompt)
            ]
            
            # 4. Call the LLM with the structured output schema for 5 articles
            # We use the 'llm' imported from llm.py
            article_set_response: ArticleSet = llm.with_structured_output(ArticleSet).invoke(messages)
            
            generated_articles: List[Article] = article_set_response.articles
            
            if not generated_articles:
                print(f"  !! WARNING: LLM generated 0 articles for '{topic}'.")
                continue

            print(f"  (Step 3/3) Successfully generated {len(generated_articles)} articles.")

            # 5. Format and append the articles to our main list
            for article_output in generated_articles:
                article_count += 1
                article_entry = {
                    "category_name": topic,
                    "article_title": article_output.article_title,
                    "article_summary": article_output.article_summary,
                    "article_body": article_output.article_body,
                    "referenced_urls": list(referenced_urls) # All articles from this batch share the same refs
                }
                all_articles_data.append(article_entry)

        except Exception as e:
            print(f"  !! FAILED to process topic '{topic}'.")
            print(f"  Error: {e}")
            # Add a placeholder entry so we know it failed
            all_articles_data.append({
                "category_name": topic,
                "article_title": f"FAILED to process topic: {topic}",
                "article_summary": f"Generation failed with error: {e}",
                "article_body": "",
                "referenced_urls": []
            })

    # 6. Save the final JSON output
    output_filename = "generated_articles.json"
    print(f"\n--- All processing complete! ---")
    print(f"Generated a total of {len(all_articles_data)} article entries.")
    
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(all_articles_data, f, indent=2, ensure_ascii=False)
        print(f"Results successfully saved to {output_filename}")
    except IOError as e:
        print(f"!! ERROR: Failed to write output file: {e}")

# 7. Run the script
if __name__ == "__main__":
    generate_all_articles()