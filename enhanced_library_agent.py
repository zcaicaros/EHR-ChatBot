"""
Enhanced Library Agent with Real PDF Processing
==============================================

This enhanced version includes actual PDF processing capabilities
and improved document search functionality.
"""

import os
import PyPDF2
import fitz  # PyMuPDF for better text extraction
from typing import List, Dict, Any, Tuple
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import numpy as np
from multiagent_system import AgentState, LIBRARY_AGENT_PROMPT

class EnhancedLibraryAgent:
    """
    Enhanced Library Agent with real PDF processing and vector search capabilities.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the enhanced library agent.
        
        Args:
            openai_api_key (str): OpenAI API key for embeddings and LLM
        """
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.1)
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        self.document_chunks = []
        
    def extract_text_from_pdf_advanced(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text from PDF using PyMuPDF for better quality extraction.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: Full text and page-by-page metadata
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            page_metadata = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # Extract additional metadata
                page_metadata.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text),
                    'word_count': len(page_text.split()),
                    'source_file': os.path.basename(pdf_path)
                })
            
            doc.close()
            return full_text, page_metadata
            
        except Exception as e:
            return f"Error reading PDF: {str(e)}", []
    
    def process_documents(self, pdf_paths: List[str]) -> None:
        """
        Process PDF documents and create vector embeddings for semantic search.
        
        Args:
            pdf_paths (List[str]): List of paths to PDF files
        """
        print("📖 Processing PDF documents for vector search...")
        
        all_chunks = []
        chunk_metadata = []
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file not found: {pdf_path}")
                continue
                
            print(f"Processing: {pdf_path}")
            
            # Extract text and metadata
            full_text, page_metadata = self.extract_text_from_pdf_advanced(pdf_path)
            
            if "Error reading PDF" in full_text:
                print(f"Error processing {pdf_path}: {full_text}")
                continue
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata.append({
                    'chunk_id': len(all_chunks) + i,
                    'source_file': os.path.basename(pdf_path),
                    'chunk_index': i,
                    'chunk_size': len(chunk)
                })
            
            all_chunks.extend(chunks)
        
        if all_chunks:
            # Create vector store
            self.vector_store = FAISS.from_texts(
                texts=all_chunks,
                embedding=self.embeddings,
                metadatas=chunk_metadata
            )
            self.document_chunks = all_chunks
            print(f"✅ Processed {len(all_chunks)} text chunks from {len(pdf_paths)} documents")
        else:
            print("❌ No documents were successfully processed")
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search on processed documents.
        
        Args:
            query (str): Search query
            k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Search results with relevance scores
        """
        if not self.vector_store:
            return []
        
        # Perform similarity search
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        search_results = []
        for doc, score in results:
            search_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': float(score),
                'source_file': doc.metadata.get('source_file', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 0)
            })
        
        return search_results
    
    def enhanced_library_search(self, state: AgentState) -> AgentState:
        """
        Enhanced library agent that uses both keyword and semantic search.
        
        Args:
            state (AgentState): Current system state
            
        Returns:
            AgentState: Updated state with comprehensive research findings
        """
        print("📚 Enhanced Library Agent: Conducting comprehensive document search...")
        
        search_strategy = state.get("search_strategy", [])
        information_requirements = state.get("information_requirements", [])
        
        # Combine search strategy and requirements for comprehensive search
        all_search_terms = search_strategy + information_requirements
        
        comprehensive_findings = []
        all_sources = set()
        
        # Perform semantic search for each search term
        for term in all_search_terms:
            if term.strip():
                print(f"🔍 Searching for: {term}")
                results = self.semantic_search(term, k=3)
                
                for result in results:
                    comprehensive_findings.append({
                        'search_term': term,
                        'content': result['content'],
                        'source': result['source_file'],
                        'relevance_score': result['relevance_score'],
                        'chunk_id': result['chunk_id']
                    })
                    all_sources.add(result['source_file'])
        
        # Prepare enhanced prompt with actual findings
        available_documents = list(all_sources) if all_sources else ["No documents processed"]
        findings_text = self._format_findings_for_llm(comprehensive_findings)
        
        enhanced_prompt = f"""
        {LIBRARY_AGENT_PROMPT}
        
        ACTUAL SEARCH RESULTS FROM DOCUMENTS:
        {findings_text}
        
        Based on these actual search results, provide your analysis and synthesis.
        """
        
        prompt = enhanced_prompt.format(
            search_strategy=search_strategy,
            information_requirements=information_requirements,
            available_documents=available_documents
        )
        
        # Get LLM analysis of the findings
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Update state with enhanced findings
        state["relevant_documents"] = [{"name": doc, "status": "processed"} for doc in all_sources]
        state["extracted_information"] = [response.content] if comprehensive_findings else ["No relevant information found in available documents"]
        state["source_references"] = list(all_sources)
        
        # Add detailed findings to state for transparency
        state["detailed_findings"] = comprehensive_findings
        
        return state
    
    def _format_findings_for_llm(self, findings: List[Dict[str, Any]]) -> str:
        """
        Format search findings for LLM processing.
        
        Args:
            findings (List[Dict[str, Any]]): Search results
            
        Returns:
            str: Formatted findings text
        """
        if not findings:
            return "No relevant information found in the available documents."
        
        formatted_text = ""
        for i, finding in enumerate(findings, 1):
            formatted_text += f"""
FINDING #{i}:
Search Term: {finding['search_term']}
Content: {finding['content'][:500]}...
Source: {finding['source']}
Relevance Score: {finding['relevance_score']:.3f}

---
"""
        return formatted_text

# Enhanced integration function
def create_enhanced_multiagent_system(api_key: str, pdf_documents: List[str] = None):
    """
    Create a multi-agent system with enhanced PDF processing capabilities.
    
    Args:
        api_key (str): OpenAI API key
        pdf_documents (List[str]): List of PDF file paths to process
        
    Returns:
        MultiAgentSystem: Enhanced system with PDF processing
    """
    from multiagent_system import MultiAgentSystem
    
    # Create enhanced library agent
    enhanced_library = EnhancedLibraryAgent(api_key)
    
    # Process PDF documents if provided
    if pdf_documents:
        enhanced_library.process_documents(pdf_documents)
    
    # Create the main system
    system = MultiAgentSystem(api_key)
    
    # Replace the library agent with enhanced version
    system.enhanced_library_agent = enhanced_library
    
    # Modify the library agent method in the system
    def enhanced_library_wrapper(state):
        return enhanced_library.enhanced_library_search(state)
    
    system.library_agent = enhanced_library_wrapper
    
    return system

# Example usage
if __name__ == "__main__":
    # Example of how to use the enhanced system
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        exit(1)
    
    # List of PDF documents to process (replace with actual paths)
    pdf_documents = [
        "example_document1.pdf",
        "example_document2.pdf"
    ]
    
    # Create enhanced system
    system = create_enhanced_multiagent_system(api_key, pdf_documents)
    
    # Example query
    user_query = "What are the key strategies for digital transformation in healthcare?"
    
    # Process the query
    results = system.process_query(user_query)
    
    # Display results
    print("\n" + "=" * 80)
    print("ENHANCED MULTI-AGENT SYSTEM RESULTS")
    print("=" * 80)
    print(f"\nQuery: {results['user_query']}")
    print(f"\nFinal Solution:\n{results['final_solution']}")
    if 'detailed_findings' in results:
        print(f"\nNumber of detailed findings: {len(results['detailed_findings'])}") 