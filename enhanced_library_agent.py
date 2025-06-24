"""
Enhanced Library Agent with Real PDF Processing
==============================================

This enhanced version includes actual PDF processing capabilities
and improved document search functionality using vector embeddings
for semantic search rather than just keyword matching.

Key Features:
- Real PDF text extraction using PyMuPDF (fitz) for better quality
- Vector embeddings for semantic document search
- FAISS vector store for fast similarity search
- Metadata tracking for source attribution
- Integration with the main multi-agent system
"""

import os
import PyPDF2
import fitz  # PyMuPDF for better text extraction than PyPDF2
from typing import List, Dict, Any, Tuple, TypedDict
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import numpy as np

# Simplified AgentState definition to avoid circular imports
class AgentState(TypedDict):
    """
    Local AgentState definition for the enhanced library agent.
    
    This avoids circular import issues while providing the necessary state structure
    for medical literature search and analysis functionality.
    """
    user_query: str                          # Original medical query
    consulting_analysis: str                 # Clinical analysis from consulting agent
    search_strategy: List[str]               # Medical literature search terms
    information_requirements: List[str]      # Specific information needed
    relevant_documents: List[Dict[str, Any]] # Processed medical documents
    extracted_information: List[str]         # Key medical information found
    source_references: List[str]             # Source citations for verification
    detailed_findings: List[Dict[str, Any]]  # Detailed search results with scores
    preliminary_solution: str                # Initial findings summary
    final_solution: str                      # Complete medical recommendations
    confidence_score: float                  # Confidence in the findings (0-1)
    messages: List[str]                      # Processing status messages

# Medical Library Agent Prompt Template
LIBRARY_AGENT_PROMPT = """
You are a Medical Research Librarian Agent specialized in evidence-based medicine and clinical literature analysis.

ROLE OVERVIEW:
Your expertise is in identifying, analyzing, and synthesizing medical literature to support evidence-based 
clinical decision-making. You search through medical documents, research papers, and clinical guidelines 
to find credible, relevant information that addresses specific medical questions.

RESEARCH METHODOLOGY:
1. SYSTEMATIC SEARCH: Use targeted medical terminology and concepts
2. EVIDENCE EVALUATION: Assess quality and relevance of sources
3. CLINICAL APPLICABILITY: Focus on actionable medical information
4. SOURCE VERIFICATION: Maintain proper citation and attribution
5. SYNTHESIS: Organize findings by clinical relevance and strength of evidence

GUIDELINES:
- Prioritize peer-reviewed medical literature and clinical guidelines
- Extract specific medical facts, protocols, and evidence-based recommendations
- Maintain accuracy - report only what is explicitly stated in sources
- Organize findings by strength of evidence and clinical applicability
- Include proper citations for medical verification and follow-up

SEARCH STRATEGY: {search_strategy}
INFORMATION REQUIREMENTS: {information_requirements}
AVAILABLE MEDICAL DOCUMENTS: {available_documents}

REQUIRED OUTPUT FORMAT:

For each relevant piece of medical evidence:

MEDICAL FINDING #[X]:
Content: [Specific medical information, guidelines, or research findings]
Source: [Document name, page/section, and publication details if available]
Evidence Level: [Research quality: High/Medium/Low based on source type]
Clinical Relevance: [Direct application to the medical query]
Recommendations: [Specific actionable medical guidance]

Focus on evidence quality and clinical applicability over quantity of findings.
"""

class EnhancedLibraryAgent:
    """
    Enhanced Library Agent with real PDF processing and vector search capabilities.
    
    This agent provides a significant upgrade over basic keyword search by:
    1. Using advanced PDF text extraction (PyMuPDF)
    2. Creating vector embeddings for semantic search
    3. Storing documents in a FAISS vector database
    4. Providing relevance scoring for search results
    
    The semantic search allows finding relevant content even when exact
    keywords don't match, making it much more powerful for research tasks.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the enhanced library agent with all necessary components.
        
        Args:
            openai_api_key (str): OpenAI API key for embeddings and LLM access
        """
        # Initialize the language model for text analysis
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.1)
        
        # Initialize embeddings model for creating vector representations of text
        # This is what enables semantic search capabilities
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        # Set up text splitter for breaking documents into manageable chunks
        # Smaller chunks improve search precision and stay within token limits
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Each chunk will be roughly 1000 characters
            chunk_overlap=200,    # 200 character overlap to maintain context between chunks
            length_function=len   # Use character count for measuring chunk size
        )
        
        # Initialize storage for the vector database and processed content
        self.vector_store = None        # Will hold the FAISS vector database
        self.document_chunks = []       # Will store the actual text chunks
        
    def extract_text_from_pdf_advanced(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text from PDF using PyMuPDF for better quality extraction.
        
        PyMuPDF (fitz) provides superior text extraction compared to PyPDF2:
        - Better handling of complex layouts
        - More accurate text positioning
        - Better support for different PDF formats
        - Preserves more formatting information
        
        Args:
            pdf_path (str): Path to the PDF file to process
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: Full extracted text and detailed page metadata
        """
        try:
            # Open the PDF document using PyMuPDF
            doc = fitz.open(pdf_path)
            full_text = ""
            page_metadata = []
            
            # Process each page individually to maintain page attribution
            for page_num in range(len(doc)):
                # Load the specific page
                page = doc.load_page(page_num)
                
                # Extract text from this page
                page_text = page.get_text()
                
                # Add page delimiter to full text for easy reference
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # Extract and store detailed metadata for each page
                # This metadata helps with source attribution and quality assessment
                page_metadata.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text),           # Helps assess content density
                    'word_count': len(page_text.split()),   # Another measure of content volume
                    'source_file': os.path.basename(pdf_path)  # Just filename, not full path
                })
            
            # Clean up memory by closing the document
            doc.close()
            
            return full_text, page_metadata
            
        except Exception as e:
            # Return error information if PDF processing fails
            # This allows the system to continue with other documents
            return f"Error reading PDF: {str(e)}", []
    
    def process_documents(self, pdf_paths: List[str]) -> None:
        """
        Process PDF documents and create vector embeddings for semantic search.
        
        This is the core preparation step that:
        1. Extracts text from all provided PDFs
        2. Splits text into optimal chunks for search
        3. Creates vector embeddings for each chunk
        4. Builds a FAISS vector database for fast search
        
        Args:
            pdf_paths (List[str]): List of paths to PDF files to process
        """
        print("📖 Processing PDF documents for vector search...")
        
        # Initialize collections for all processed content
        all_chunks = []          # Will store all text chunks from all documents
        chunk_metadata = []      # Will store metadata for each chunk
        
        # Process each PDF document individually
        for pdf_path in pdf_paths:
            # Verify the file exists before trying to process it
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file not found: {pdf_path}")
                continue
                
            print(f"Processing: {pdf_path}")
            
            # Extract text and metadata from this PDF
            full_text, page_metadata = self.extract_text_from_pdf_advanced(pdf_path)
            
            # Skip this document if extraction failed
            if "Error reading PDF" in full_text:
                print(f"Error processing {pdf_path}: {full_text}")
                continue
            
            # Split the full document text into smaller, searchable chunks
            # This improves search precision and manages token limits
            chunks = self.text_splitter.split_text(full_text)
            
            # Create detailed metadata for each chunk to enable source tracking
            for i, chunk in enumerate(chunks):
                chunk_metadata.append({
                    'chunk_id': len(all_chunks) + i,              # Unique identifier across all documents
                    'source_file': os.path.basename(pdf_path),    # Source document name
                    'chunk_index': i,                             # Position within this document
                    'chunk_size': len(chunk)                      # Size for quality assessment
                })
            
            # Add this document's chunks to the overall collection
            all_chunks.extend(chunks)
        
        # Create the vector database if we successfully processed any documents
        if all_chunks:
            # Create FAISS vector store with embeddings
            # FAISS is a highly optimized library for similarity search
            self.vector_store = FAISS.from_texts(
                texts=all_chunks,              # The actual text content
                embedding=self.embeddings,     # OpenAI embeddings model
                metadatas=chunk_metadata       # Metadata for each chunk
            )
            
            # Store chunks for reference (useful for debugging or display)
            self.document_chunks = all_chunks
            
            print(f"✅ Processed {len(all_chunks)} text chunks from {len(pdf_paths)} documents")
        else:
            print("❌ No documents were successfully processed")
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search on processed documents.
        
        This uses vector similarity to find relevant content, which is much more
        powerful than keyword search because it understands meaning and context.
        
        For example, searching for "digital transformation" might also find content
        about "technology modernization" or "digital innovation" even if those
        exact terms aren't used.
        
        Args:
            query (str): Search query (can be natural language)
            k (int): Number of top results to return (default: 5)
            
        Returns:
            List[Dict[str, Any]]: Search results with relevance scores and metadata
        """
        # Return empty results if no vector store has been created
        if not self.vector_store:
            return []
        
        # Perform similarity search using vector embeddings
        # This computes the cosine similarity between the query embedding
        # and all document chunk embeddings
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format results into a standardized structure
        search_results = []
        for doc, score in results:
            search_results.append({
                'content': doc.page_content,                    # The actual text content
                'metadata': doc.metadata,                       # Source and positioning info
                'relevance_score': float(score),                # Similarity score (lower is better)
                'source_file': doc.metadata.get('source_file', 'Unknown'),  # Source document
                'chunk_id': doc.metadata.get('chunk_id', 0)     # Unique chunk identifier
            })
        
        return search_results
    
    def enhanced_library_search(self, state: AgentState) -> AgentState:
        """
        Enhanced library agent that uses both keyword and semantic search.
        
        This is the main entry point that integrates with the multi-agent system.
        It takes the consulting agent's strategy and performs comprehensive
        document search using vector embeddings.
        
        Args:
            state (AgentState): Current system state from previous agents
            
        Returns:
            AgentState: Updated state with comprehensive research findings
        """
        print("📚 Enhanced Library Agent: Conducting comprehensive document search...")
        
        # Extract search strategy from the consulting agent's work
        search_strategy = state.get("search_strategy", [])
        information_requirements = state.get("information_requirements", [])
        
        # Combine both types of search terms for comprehensive coverage
        # This ensures we search for both specific terms and general information types
        all_search_terms = search_strategy + information_requirements
        
        # Initialize collections for all findings
        comprehensive_findings = []     # Detailed search results
        all_sources = set()            # Unique source documents (using set to avoid duplicates)
        
        # Perform semantic search for each search term identified by consulting agent
        for term in all_search_terms:
            if term.strip():  # Only search non-empty terms
                print(f"🔍 Searching for: {term}")
                
                # Perform semantic search for this specific term
                results = self.semantic_search(term, k=3)  # Get top 3 results per term
                
                # Process and store each search result
                for result in results:
                    comprehensive_findings.append({
                        'search_term': term,                          # What we searched for
                        'content': result['content'],                 # The found content
                        'source': result['source_file'],              # Source document
                        'relevance_score': result['relevance_score'], # How relevant it is
                        'chunk_id': result['chunk_id']                # Unique identifier
                    })
                    
                    # Track unique sources for summary reporting
                    all_sources.add(result['source_file'])
        
        # Prepare information for the language model analysis
        available_documents = list(all_sources) if all_sources else ["No documents processed"]
        findings_text = self._format_findings_for_llm(comprehensive_findings)
        
        # Create enhanced prompt that includes actual search results
        # This gives the LLM real data to work with instead of simulated research
        enhanced_prompt = f"""
        {LIBRARY_AGENT_PROMPT}
        
        ACTUAL SEARCH RESULTS FROM DOCUMENTS:
        {findings_text}
        
        Based on these actual search results, provide your analysis and synthesis.
        """
        
        # Format the prompt with all necessary information
        prompt = enhanced_prompt.format(
            search_strategy=search_strategy,
            information_requirements=information_requirements,
            available_documents=available_documents
        )
        
        # Get LLM analysis of the findings
        # The LLM will interpret and synthesize the search results
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Update state with enhanced findings and analysis
        state["relevant_documents"] = [{"name": doc, "status": "processed"} for doc in all_sources]
        state["extracted_information"] = [response.content] if comprehensive_findings else ["No relevant information found in available documents"]
        state["source_references"] = list(all_sources)
        
        # Add detailed findings to state for transparency and debugging
        # This allows other agents or users to see exactly what was found
        state["detailed_findings"] = comprehensive_findings
        
        return state
    
    def _format_findings_for_llm(self, findings: List[Dict[str, Any]]) -> str:
        """
        Format search findings for LLM processing.
        
        This function takes the raw search results and formats them in a way
        that's easy for the language model to understand and analyze.
        
        Args:
            findings (List[Dict[str, Any]]): Raw search results from vector search
            
        Returns:
            str: Formatted text that's optimized for LLM processing
        """
        # Handle the case where no relevant information was found
        if not findings:
            return "No relevant information found in the available documents."
        
        # Format each finding in a structured way
        formatted_text = ""
        for i, finding in enumerate(findings, 1):
            # Truncate content to keep prompt manageable while preserving key information
            content_preview = finding['content'][:500]  # First 500 characters
            if len(finding['content']) > 500:
                content_preview += "..."  # Indicate there's more content
            
            # Create a structured entry for each finding
            formatted_text += f"""
            FINDING #{i}:
            Search Term: {finding['search_term']}
            Content: {content_preview}
            Source: {finding['source']}
            Relevance Score: {finding['relevance_score']:.3f}

            ---
            """
        return formatted_text