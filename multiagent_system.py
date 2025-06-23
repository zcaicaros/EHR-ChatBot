"""
Multi-Agent System using LangGraph
==================================

This system consists of three specialized agents:
1. Consulting Agent: Analyzes user queries and plans the information gathering strategy
2. Library Agent: Searches through PDF documents to find relevant information
3. Solution Agent: Synthesizes information to provide comprehensive solutions

Author: Cong Zhang
"""

import os
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import PyPDF2
import re
from datetime import datetime

from enhanced_library_agent import EnhancedLibraryAgent

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    Represents the state that flows through our multi-agent system.
    Each agent can read from and write to this shared state.
    
    This acts as a shared memory system where:
    - Each agent contributes their specialized knowledge
    - Information flows from one agent to the next
    - All agents can access previous work for context
    """
    # User's original query - the starting point for all processing
    user_query: str
    
    # Consulting agent's analysis and strategy
    # These fields capture the strategic thinking and planning phase
    consulting_analysis: str              # Deep analysis of what the user really needs
    search_strategy: List[str]           # Specific terms and concepts to search for
    
    # Library agent's findings
    # These fields store the research results and source tracking
    relevant_documents: List[Dict[str, Any]]  # Documents that were found and processed
    extracted_information: List[str]          # Key information extracted from documents
    source_references: List[str]             # Citations and references for verification
    detailed_findings: List[Dict[str, Any]]   # Enhanced: Detailed search results with scores
    
    # Solution agent's work
    # These fields contain the final synthesis and recommendations
    preliminary_solution: str            # Initial draft solution (currently unused)
    final_solution: str                 # Complete, polished answer to user query
    confidence_score: float             # How confident the system is in the solution (0-1)
    
    # System messages for debugging/tracking
    # Helps track the processing flow and timing for debugging
    messages: List[str]

# ============================================================================
# AGENT PROMPTS
# ============================================================================

# Each prompt is carefully designed to guide the AI agent's thinking process
# and ensure consistent, high-quality outputs

CONSULTING_AGENT_PROMPT = """
You are a Senior Consulting Agent with expertise in problem analysis and strategic thinking.

Your role is to:
1. DEEPLY ANALYZE the user's query to understand the underlying needs and context
2. IDENTIFY what specific information would be most valuable to answer their question
3. DEVELOP a strategic approach for information gathering

Guidelines for your analysis:
- Think beyond the surface level of the question
- Consider multiple perspectives and potential edge cases
- Identify key concepts, entities, and relationships that need to be researched
- Prioritize information needs based on relevance and importance
- Be thorough but focused

User Query: {user_query}

Provide your analysis in the following format:

ANALYSIS:
[Your deep thinking about the query, including context, implications, and underlying needs]

SEARCH STRATEGY:
[List 3-5 specific search terms or concepts that should be investigated]

Be thorough and strategic in your thinking.
"""

LIBRARY_AGENT_PROMPT = """
You are a Research Librarian Agent specialized in document analysis and information extraction.

Your role is to:
1. SEARCH through provided documents using the consulting agent's strategy
2. EXTRACT relevant information that directly addresses the user's needs
3. IDENTIFY credible sources and maintain proper attribution
4. SYNTHESIZE findings in a clear, organized manner

Guidelines for your research:
- Focus on information that directly relates to the search strategy
- Extract specific facts, data, quotes, and references
- Maintain accuracy and avoid interpretation - stick to what's explicitly stated
- Organize findings by relevance and credibility
- Include page numbers and source references for verification

Search Strategy: {search_strategy}

Available Documents: {available_documents}

For each piece of relevant information you find, provide:

FINDING #:
Content: [Exact text or paraphrased information]
Source: [Document name and page/section reference]
Relevance: [How this relates to the user's query]
Confidence: [High/Medium/Low based on source credibility]

Focus on quality over quantity - better to have fewer, highly relevant findings.
"""

SOLUTION_AGENT_PROMPT = """
You are a Solution Synthesis Agent with expertise in combining research findings into actionable insights.

Your role is to:
1. ANALYZE all information gathered by the library agent
2. SYNTHESIZE findings into a coherent, comprehensive solution
3. ADDRESS the user's original query directly and thoroughly
4. PROVIDE actionable recommendations when appropriate

Guidelines for your solution:
- Start with a direct answer to the user's query
- Support your answer with evidence from the research
- Acknowledge any limitations or gaps in the available information
- Provide clear, actionable next steps when applicable
- Maintain objectivity while being helpful

User Query: {user_query}
Consulting Analysis: {consulting_analysis}
Research Findings: {extracted_information}
Source References: {source_references}

Structure your response as follows:

DIRECT ANSWER:
[Clear, direct response to the user's query]

SUPPORTING EVIDENCE:
[Key findings that support your answer, with source references]

ANALYSIS:
[Your synthesis of the information and any insights derived]

RECOMMENDATIONS:
[Actionable next steps or suggestions, if applicable]

LIMITATIONS:
[Any gaps in information or caveats the user should be aware of]

CONFIDENCE ASSESSMENT:
[Rate your confidence in this solution from 1-10 and explain why]

Be comprehensive yet concise, and always prioritize accuracy over completeness.
"""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file using PyPDF2.
    
    This function handles the low-level PDF processing and provides
    basic error handling for common PDF issues.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content with page markers, or error message
    """
    try:
        # Open the PDF file in binary read mode
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Process each page sequentially
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                
                # Add page markers to help with source attribution later
                text += f"\n--- Page {page_num + 1} ---\n"
                
                # Extract text from the current page
                text += page.extract_text()
                
        return text
    except Exception as e:
        # Return a descriptive error message if PDF processing fails
        return f"Error reading PDF: {str(e)}"

def search_text_for_keywords(text: str, keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Search for keywords in text and return relevant excerpts with context.
    
    This function performs basic keyword matching and extracts surrounding
    context to help understand the relevance of each finding.
    
    Args:
        text (str): Text to search through
        keywords (List[str]): Keywords to search for
        
    Returns:
        List[Dict[str, Any]]: List of relevant excerpts with context and metadata
    """
    findings = []
    text_lower = text.lower()  # Convert to lowercase for case-insensitive search
    
    # Search for each keyword individually
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Only proceed if the keyword is actually found in the text
        if keyword_lower in text_lower:
            # Find all occurrences of this keyword
            start = 0
            while True:
                # Search for the next occurrence of the keyword
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:  # No more occurrences found
                    break
                
                # Extract context around the keyword (200 chars before and after)
                # This provides enough context to understand the usage
                context_start = max(0, pos - 200)
                context_end = min(len(text), pos + len(keyword) + 200)
                context = text[context_start:context_end].strip()
                
                # Store the finding with metadata for later processing
                findings.append({
                    'keyword': keyword,
                    'context': context,
                    'position': pos,
                    'relevance_score': len(keyword)  # Simple scoring based on keyword length
                })
                
                # Move start position to find additional occurrences
                start = pos + 1
    
    # Sort findings by relevance score (longer keywords are considered more specific)
    findings.sort(key=lambda x: x['relevance_score'], reverse=True)
    return findings

# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class MultiAgentSystem:
    """
    Main class that orchestrates the multi-agent system using LangGraph.
    
    This class sets up the workflow where agents process information sequentially:
    Consulting Agent → Enhanced Library Agent → Solution Agent
    
    Each agent specializes in a different aspect of problem-solving, creating
    a pipeline that mimics how human experts might collaborate.
    The Enhanced Library Agent provides real PDF processing with vector search.
    """
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4"):
        """
        Initialize the multi-agent system with OpenAI integration and enhanced library capabilities.
        
        Args:
            openai_api_key (str): OpenAI API key for accessing GPT models
            model_name (str): OpenAI model to use (default: gpt-4 for best results)
        """
        # Set up the language model with optimal settings for this use case
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1  # Low temperature for consistent, focused responses
        )
        
        # Initialize the enhanced library agent with real PDF processing capabilities
        self.enhanced_library_agent = EnhancedLibraryAgent(openai_api_key)
        
        # Build the agent workflow graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow that defines how agents interact.
        
        This creates a directed graph where information flows from one agent
        to the next, with each agent building upon the previous agent's work.
        
        Returns:
            StateGraph: Configured LangGraph workflow ready for execution
        """
        # Create the state graph that will manage agent interactions
        workflow = StateGraph(AgentState)
        
        # Add each agent as a node in the workflow
        # Each node represents a processing step in our pipeline
        workflow.add_node("consulting_agent", self.consulting_agent)
        workflow.add_node("library_agent", self.library_agent)
        workflow.add_node("solution_agent", self.solution_agent)
        
        # Define the workflow edges (how agents connect)
        # This creates a linear pipeline: consulting → library → solution → end
        workflow.add_edge("consulting_agent", "library_agent")
        workflow.add_edge("library_agent", "solution_agent")
        workflow.add_edge("solution_agent", END)
        
        # Set the entry point - all processing starts with the consulting agent
        workflow.set_entry_point("consulting_agent")
        
        # Compile the workflow into an executable graph
        return workflow.compile()
    
    def consulting_agent(self, state: AgentState) -> AgentState:
        """
        Consulting Agent: Analyzes user query and develops information gathering strategy.
        
        This agent acts like a senior consultant who:
        1. Understands what the user really needs (not just what they ask)
        2. Plans the research strategy
        
        Args:
            state (AgentState): Current system state containing user query
            
        Returns:
            AgentState: Updated state with consulting analysis and strategy
        """
        print("🧠 Consulting Agent: Analyzing query and developing strategy...")
        
        # Prepare the prompt by inserting the user's query
        prompt = CONSULTING_AGENT_PROMPT.format(
            user_query=state["user_query"]
        )
        
        # Get response from the LLM using the structured prompt
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Parse the response to extract structured information
        # The LLM returns formatted text that we need to break down into components
        response_text = response.content
        
        # Extract the analysis section using regex pattern matching
        analysis_match = re.search(r'ANALYSIS:\s*(.*?)\s*SEARCH STRATEGY:', response_text, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else response_text
        
        # Extract search strategy items
        strategy_match = re.search(r'SEARCH STRATEGY:\s*(.*?)$', response_text, re.DOTALL)
        search_strategy = []
        if strategy_match:
            strategy_text = strategy_match.group(1).strip()
            # Split by lines and clean up each item
            search_strategy = [item.strip() for item in strategy_text.split('\n') if item.strip()]
        
        # Update the shared state with our findings
        # This information will be available to subsequent agents
        state["consulting_analysis"] = analysis
        state["search_strategy"] = search_strategy
        
        # Add a timestamp message for debugging and tracking
        state["messages"].append(f"Consulting Agent completed analysis at {datetime.now()}")
        
        return state
    
    def library_agent(self, state: AgentState) -> AgentState:
        """
        Enhanced Library Agent: Uses real PDF processing and vector search for document analysis.
        
        This agent acts like a research librarian with advanced tools who:
        1. Takes the consulting agent's strategy
        2. Performs semantic search through processed PDF documents
        3. Extracts relevant information with proper citations and relevance scores
        4. Provides detailed findings with source attribution
        
        Args:
            state (AgentState): Current system state with search strategy
            
        Returns:
            AgentState: Updated state with comprehensive research findings and source references
        """
        print("📚 Enhanced Library Agent: Conducting semantic search through documents...")
        
        # Use the enhanced library agent for real document processing
        # This provides vector-based semantic search instead of simulated research
        enhanced_state = self.enhanced_library_agent.enhanced_library_search(state)
        
        # The enhanced library agent updates the state with:
        # - relevant_documents: List of processed documents with metadata
        # - extracted_information: LLM analysis of search results
        # - source_references: List of source documents
        # - detailed_findings: Raw search results with relevance scores
        
        # Add tracking message for the enhanced processing
        enhanced_state["messages"].append(f"Enhanced Library Agent completed semantic search at {datetime.now()}")
        
        return enhanced_state
    
    def solution_agent(self, state: AgentState) -> AgentState:
        """
        Solution Agent: Synthesizes information into a comprehensive solution.
        
        This agent acts like a senior analyst who:
        1. Reviews all the research findings
        2. Synthesizes information into a coherent answer
        3. Provides actionable recommendations
        4. Assesses confidence in the solution
        
        Args:
            state (AgentState): Current system state with all research findings
            
        Returns:
            AgentState: Updated state with final solution and confidence score
        """
        print("💡 Solution Agent: Synthesizing information into final solution...")
        
        # Prepare comprehensive prompt with all collected information
        # Include detailed findings if available from enhanced library agent
        detailed_findings_summary = ""
        if state.get("detailed_findings"):
            detailed_findings_summary = f"\n\nDETAILED SEARCH FINDINGS:\n"
            for i, finding in enumerate(state["detailed_findings"][:5], 1):  # Include top 5 findings
                detailed_findings_summary += f"{i}. Search term: '{finding['search_term']}' | "
                detailed_findings_summary += f"Source: {finding['source']} | "
                detailed_findings_summary += f"Relevance: {finding['relevance_score']:.3f}\n"
                detailed_findings_summary += f"   Content: {finding['content'][:200]}...\n\n"
        
        prompt = SOLUTION_AGENT_PROMPT.format(
            user_query=state["user_query"],
            consulting_analysis=state["consulting_analysis"],
            extracted_information="\n".join(state["extracted_information"]) + detailed_findings_summary,
            source_references=", ".join(state["source_references"])
        )
        
        # Get the final synthesized response from LLM
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        response_text = response.content
        
        # Extract confidence score from the response
        # Look for a number in the confidence assessment section
        confidence_match = re.search(r'CONFIDENCE ASSESSMENT:\s*.*?(\d+)', response_text)
        if confidence_match:
            # Convert from 1-10 scale to 0-1 scale
            confidence_score = float(confidence_match.group(1)) / 10
        else:
            # Default confidence if no score is found
            confidence_score = 0.8
        
        # Update state with final results
        state["final_solution"] = response_text
        state["confidence_score"] = confidence_score
        
        # Add completion tracking message
        state["messages"].append(f"Solution Agent completed synthesis at {datetime.now()}")
        
        return state
    
    def process_query(self, user_query: str, pdf_documents: List[str] = None) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system with enhanced PDF processing.
        
        This is the main entry point that orchestrates the entire pipeline:
        1. Processes any provided PDF documents into vector embeddings
        2. Initializes the shared state
        3. Runs the query through all agents in sequence
        4. Returns the complete results with enhanced document analysis
        
        Args:
            user_query (str): The user's question or request
            pdf_documents (List[str]): List of PDF file paths to search through (optional)
            
        Returns:
            Dict[str, Any]: Complete results from all agents including final solution
        """
        print(f"🚀 Starting enhanced multi-agent processing for query: '{user_query}'")
        print("=" * 80)
        
        # Process PDF documents if provided
        if pdf_documents:
            print(f"📄 Processing {len(pdf_documents)} PDF documents for semantic search...")
            self.enhanced_library_agent.process_documents(pdf_documents)
            print("✅ PDF processing completed!")
        else:
            print("📄 No PDF documents provided - using any previously processed documents")
        
        # Initialize the shared state that will flow through all agents
        # All fields start empty except for the user query and messages list
        initial_state = AgentState(
            user_query=user_query,
            consulting_analysis="",
            search_strategy=[],
            relevant_documents=[],
            extracted_information=[],
            source_references=[],
            detailed_findings=[],  # Initialize enhanced field for detailed search results
            preliminary_solution="",
            final_solution="",
            confidence_score=0.0,
            messages=[]  # Initialize empty list for tracking messages
        )
        
        # Process the query through the entire agent pipeline
        # The graph handles the sequential execution: consulting → enhanced_library → solution
        final_state = self.graph.invoke(initial_state)
        
        print("=" * 80)
        print("✅ Enhanced multi-agent processing completed!")
        
        # Return the final state which contains all agent outputs
        return final_state

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """
    Example usage of the enhanced multi-agent system with PDF processing.
    
    This function demonstrates how to:
    1. Initialize the system with API credentials
    2. Specify PDF documents to process (optional)
    3. Process a sample query with real document search
    4. Display comprehensive results including detailed findings
    """
    # Initialize the system (requires OpenAI API key in environment)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create the enhanced multi-agent system instance
    system = MultiAgentSystem(api_key)
    
    # Example PDF documents to process (replace with actual PDF paths)
    # These should be paths to real PDF files you want to search through
    pdf_documents = [
        # "path/to/your/document1.pdf",  # Uncomment and replace with actual paths
        # "path/to/your/document2.pdf",  # Add as many documents as needed
        "./ehab368.pdf"
    ]
    
    # Example query - replace with any question you want to research
    user_query = "Can you show me some tips on how to treat acute and chronic heart failure?"
    
    # Process the query through all agents with enhanced PDF processing
    results = system.process_query(user_query, pdf_documents)
    
    # Display comprehensive results in a structured format
    print("\n" + "=" * 80)
    print("ENHANCED MULTI-AGENT SYSTEM RESULTS")
    print("=" * 80)
    print(f"\nOriginal Query: {results['user_query']}")
    print(f"\nConsulting Analysis:\n{results['consulting_analysis']}")
    print(f"\nSearch Strategy: {results['search_strategy']}")
    print(f"\nFinal Solution:\n{results['final_solution']}")
    print(f"\nConfidence Score: {results['confidence_score']}")
    
    # Display enhanced information if available
    if results.get('detailed_findings'):
        print(f"\nNumber of detailed search findings: {len(results['detailed_findings'])}")
        
        # Show a sample of the detailed findings for transparency
        if results['detailed_findings']:
            print("\nSample of detailed findings:")
            for i, finding in enumerate(results['detailed_findings'][:], 1):
                print(f"\n  Finding {i}:")
                print(f"    Search term: {finding['search_term']}")
                print(f"    Source: {finding['source']}")
                print(f"    Relevance: {finding['relevance_score']:.3f}")
                print(f"    Content preview: {finding['content'][:]}")
    
    # Display source information
    if results.get('source_references'):
        print(f"\nSource documents processed: {', '.join(results['source_references'])}")
    
    print("\n" + "=" * 80)
    print("✅ Enhanced processing demonstration completed!")

# Entry point for running the script directly
if __name__ == "__main__":
    main() 