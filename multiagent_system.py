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
from patient_scanning_agent import PatientQueryAgent

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
    
    # Patient context information
    patient_id: str                      # Patient ID if query is patient-specific
    
    # Consulting agent's analysis and strategy
    # These fields capture the strategic thinking and planning phase
    consulting_analysis: str              # Deep analysis of what the user really needs
    search_strategy: List[str]           # Specific terms and concepts to search for library
    patient_queries: List[str]           # Specific queries for patient data
    
    # Library agent's findings
    # These fields store the research results and source tracking
    relevant_documents: List[Dict[str, Any]]  # Documents that were found and processed
    extracted_information: List[str]          # Key information extracted from documents
    source_references: List[str]             # Citations and references for verification
    detailed_findings: List[Dict[str, Any]]   # Enhanced: Detailed search results with scores
    
    # Patient agent's findings
    # These fields store patient-specific data and analysis
    patient_data: Dict[str, Any]         # Patient data retrieved from records
    patient_analysis: List[str]          # Analysis of patient-specific information
    patient_findings: List[Dict[str, Any]]  # Detailed patient data findings
    
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
You are a Senior Medical Consulting Agent with expertise in problem analysis and strategic thinking for patient care and medical research.

Your role is to:
1. DEEPLY ANALYZE the user's query to understand the underlying needs and context
2. IDENTIFY whether the query requires patient-specific data, general medical research, or both
3. DEVELOP specific queries for patient data extraction and library research
4. DETERMINE the patient ID if the query is patient-specific

Guidelines for your analysis:
- Think beyond the surface level of the question
- Consider both patient-specific information needs and general medical knowledge requirements
- Identify key concepts, entities, and relationships that need to be researched
- Separate patient data needs from general research needs
- Be thorough but focused

User Query: {user_query}
Patient ID (if provided): {patient_id}

Provide your analysis in the following format:

ANALYSIS:
[Your deep thinking about the query, including context, implications, and underlying needs]

PATIENT QUERIES:
[List 2-4 specific questions about patient data needed to answer the query, or "None" if no patient data needed]

LIBRARY SEARCH STRATEGY:
[List 2-4 specific search terms or concepts for medical literature/documents, or "None" if no library research needed]

Be thorough and strategic in your thinking.
"""

PATIENT_AGENT_PROMPT = """
You are a Patient Data Analyst Agent specialized in extracting and analyzing patient medical records.

Your role is to:
1. PROCESS specific patient queries using the patient database
2. EXTRACT relevant patient information that directly addresses the user's needs
3. ANALYZE patient data patterns and provide medical insights
4. MAINTAIN patient privacy while providing comprehensive analysis

Guidelines for your analysis:
- Focus on patient data that directly relates to the patient queries
- Extract specific medical facts, values, dates, and trends
- Provide context for medical findings when possible
- Organize findings by medical relevance and chronology
- Include relevant medical codes and references

Patient Queries: {patient_queries}
Patient ID: {patient_id}

For each query, provide:

PATIENT FINDING #:
Query: [The specific patient query being addressed]
Data Found: [Relevant patient data extracted]
Analysis: [Medical context and interpretation]
Significance: [How this relates to the overall patient assessment]

Focus on accuracy and medical relevance.
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
You are a Medical Solution Synthesis Agent with expertise in combining patient data and research findings into comprehensive medical insights.

Your role is to:
1. ANALYZE all information gathered from both patient records and medical literature
2. SYNTHESIZE findings into a coherent, comprehensive medical solution
3. ADDRESS the user's original query directly and thoroughly
4. PROVIDE actionable medical recommendations when appropriate

Guidelines for your solution:
- Start with a direct answer to the user's query
- Integrate patient-specific data with general medical knowledge
- Support your answer with evidence from both patient records and research
- Acknowledge any limitations or gaps in the available information
- Provide clear, actionable medical recommendations when applicable
- Maintain patient privacy and professional medical standards

User Query: {user_query}
Patient ID: {patient_id}
Consulting Analysis: {consulting_analysis}

Patient Information:
{patient_analysis}

Research Findings:
{extracted_information}

Source References: {source_references}

Structure your response as follows:

DIRECT ANSWER:
[Clear, direct response to the user's query]

PATIENT-SPECIFIC INSIGHTS:
[Key findings from patient data analysis]

SUPPORTING MEDICAL EVIDENCE:
[Key findings from medical literature with source references]

INTEGRATED ANALYSIS:
[Your synthesis combining patient data with research findings]

MEDICAL RECOMMENDATIONS:
[Actionable medical recommendations based on integrated analysis]

LIMITATIONS:
[Any gaps in information or caveats the user should be aware of]

CONFIDENCE ASSESSMENT:
[Rate your confidence in this solution from 1-10 and explain why]

Be comprehensive yet concise, and always prioritize accuracy over completeness.
"""

# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class MultiAgentSystem:
    """
    Main class that orchestrates the multi-agent system using LangGraph.
    
    This class sets up the workflow where agents process information in parallel:
    Consulting Agent → [Patient Agent + Library Agent] → Solution Agent
    
    Each agent specializes in a different aspect of problem-solving, creating
    a pipeline that mimics how human medical experts might collaborate.
    The Patient Agent handles patient-specific data queries while the 
    Enhanced Library Agent provides general medical research capabilities.
    """
    
    def __init__(self, openai_api_key: str, patients_json_path: str = './patients.json', model_name: str = "gpt-4"):
        """
        Initialize the multi-agent system with OpenAI integration, patient data, and enhanced library capabilities.
        
        Args:
            openai_api_key (str): OpenAI API key for accessing GPT models
            patients_json_path (str): Path to the patients JSON file
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
        
        # Initialize the patient query agent with AI capabilities
        self.patient_query_agent = PatientQueryAgent(
            json_file_path=patients_json_path,
            openai_api_key=openai_api_key,
            model_name=model_name
        )
        
        # Build the agent workflow graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow that defines how agents interact.
        
        This creates a directed graph with proper sequential routing:
        Consulting Agent → Patient Agent (if needed) → Library Agent (if needed) → Solution Agent
        
        Returns:
            StateGraph: Configured LangGraph workflow ready for execution
        """
        # Create the state graph that will manage agent interactions
        workflow = StateGraph(AgentState)
        
        # Add each agent as a node in the workflow
        workflow.add_node("consulting_agent", self.consulting_agent)
        workflow.add_node("patient_agent", self.patient_agent)
        workflow.add_node("library_agent", self.library_agent)
        workflow.add_node("solution_agent", self.solution_agent)
        
        # Define routing after consulting agent
        def route_after_consulting(state: AgentState):
            """Route to first required agent or directly to solution."""
            patient_queries = state.get("patient_queries", [])
            search_strategy = state.get("search_strategy", [])
            
            # Priority: patient agent first, then library agent
            if patient_queries and any(q.lower() != "none" for q in patient_queries):
                return "patient_agent"
            elif search_strategy and any(s.lower() != "none" for s in search_strategy):
                return "library_agent"
            else:
                return "solution_agent"
        
        # Define routing after patient agent
        def route_after_patient(state: AgentState):
            """Route to library agent if needed, otherwise to solution."""
            search_strategy = state.get("search_strategy", [])
            
            if search_strategy and any(s.lower() != "none" for s in search_strategy):
                return "library_agent"
            else:
                return "solution_agent"
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "consulting_agent",
            route_after_consulting,
            {
                "patient_agent": "patient_agent",
                "library_agent": "library_agent", 
                "solution_agent": "solution_agent"
            }
        )
        
        workflow.add_conditional_edges(
            "patient_agent",
            route_after_patient,
            {
                "library_agent": "library_agent",
                "solution_agent": "solution_agent"
            }
        )
        
        # Library agent always goes to solution
        workflow.add_edge("library_agent", "solution_agent")
        workflow.add_edge("solution_agent", END)
        
        # Set the entry point
        workflow.set_entry_point("consulting_agent")
        
        # Compile the workflow into an executable graph
        return workflow.compile()
    
    def consulting_agent(self, state: AgentState) -> AgentState:
        """
        Consulting Agent: Analyzes user query and develops information gathering strategy.
        
        This agent acts like a senior medical consultant who:
        1. Understands what the user really needs (not just what they ask)
        2. Plans both patient data queries and research strategy
        3. Routes to appropriate specialized agents
        
        Args:
            state (AgentState): Current system state containing user query
            
        Returns:
            AgentState: Updated state with consulting analysis and strategy
        """
        print("🧠 Consulting Agent: Analyzing query and developing strategy...")
        
        # Prepare the prompt by inserting the user's query and patient ID
        prompt = CONSULTING_AGENT_PROMPT.format(
            user_query=state["user_query"],
            patient_id=state.get("patient_id", "Not provided")
        )
        
        # Get response from the LLM using the structured prompt
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Parse the response to extract structured information
        response_text = response.content
        
        # Extract the analysis section
        analysis_match = re.search(r'ANALYSIS:\s*(.*?)\s*PATIENT QUERIES:', response_text, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else response_text
        
        # Extract patient queries
        patient_queries_match = re.search(r'PATIENT QUERIES:\s*(.*?)\s*LIBRARY SEARCH STRATEGY:', response_text, re.DOTALL)
        patient_queries = []
        if patient_queries_match:
            queries_text = patient_queries_match.group(1).strip()
            if queries_text.lower() != "none":
                patient_queries = [item.strip() for item in queries_text.split('\n') if item.strip()]
        
        # Extract library search strategy
        strategy_match = re.search(r'LIBRARY SEARCH STRATEGY:\s*(.*?)$', response_text, re.DOTALL)
        search_strategy = []
        if strategy_match:
            strategy_text = strategy_match.group(1).strip()
            if strategy_text.lower() != "none":
                                search_strategy = [item.strip() for item in strategy_text.split('\n') if item.strip()]
        
        # Update the shared state with our findings
        state["consulting_analysis"] = analysis
        state["patient_queries"] = patient_queries
        state["search_strategy"] = search_strategy
        
        # Add a timestamp message for debugging and tracking
        state["messages"].append(f"Consulting Agent completed analysis at {datetime.now()}")
        
        return state

    def patient_agent(self, state: AgentState) -> AgentState:
        """
        Patient Agent: Processes patient-specific queries using the PatientQueryAgent.
        
        This agent acts like a medical records specialist who:
        1. Takes specific patient queries from the consulting agent
        2. Extracts relevant patient data from medical records
        3. Provides medical analysis and context
        
        Args:
            state (AgentState): Current system state with patient queries
            
        Returns:
            AgentState: Updated state with patient data and analysis
        """
        print("👤 Patient Agent: Analyzing patient records...")
        
        patient_queries = state.get("patient_queries", [])
        patient_id = state.get("patient_id")
        
        if not patient_queries or not patient_id:
            print("   No patient queries or patient ID provided, skipping patient analysis")
            state["patient_data"] = {}
            state["patient_analysis"] = []
            state["patient_findings"] = []
            state["messages"].append(f"Patient Agent: No patient queries or ID at {datetime.now()}")
            return state
        
        # Process each patient query using the PatientQueryAgent
        patient_analysis = []
        patient_findings = []
        all_patient_data = {}
        
        for i, query in enumerate(patient_queries):
            print(f"   Processing patient query {i+1}: {query}")
            
            try:
                # Use the PatientQueryAgent to process the query
                result = self.patient_query_agent.process_query(query, patient_id)
                
                # Store the structured response
                finding = {
                    "query": query,
                    "patient_id": patient_id,
                    "answer": result.get("answer", "No answer available"),
                    "query_type": result.get("query_type", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "relevant_data": result.get("relevant_data", {})
                }
                patient_findings.append(finding)
                
                # Merge relevant data into all_patient_data
                if result.get("relevant_data"):
                    for data_type, data in result["relevant_data"].items():
                        if data_type not in all_patient_data:
                            all_patient_data[data_type] = data
                        elif isinstance(data, list) and isinstance(all_patient_data[data_type], list):
                            all_patient_data[data_type].extend(data)
                
                # Generate analysis text using LLM
                analysis_prompt = PATIENT_AGENT_PROMPT.format(
                    patient_queries=[query],
                    patient_id=patient_id
                )
                
                # Add the patient data context to the prompt
                analysis_prompt += f"\n\nPatient Data Retrieved:\n{result.get('answer', 'No data')}"
                
                messages = [SystemMessage(content=analysis_prompt)]
                analysis_response = self.llm.invoke(messages)
                patient_analysis.append(analysis_response.content)
                
            except Exception as e:
                print(f"   Error processing patient query: {e}")
                patient_analysis.append(f"Error processing query '{query}': {str(e)}")
        
        # Update state with patient findings
        state["patient_data"] = all_patient_data
        state["patient_analysis"] = patient_analysis
        state["patient_findings"] = patient_findings
        
        # Add tracking message
        state["messages"].append(f"Patient Agent completed analysis of {len(patient_queries)} queries at {datetime.now()}")
        
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
        Solution Agent: Synthesizes information into a comprehensive medical solution.
        
        This agent acts like a senior medical analyst who:
        1. Reviews all the patient data and research findings
        2. Synthesizes information into a coherent medical answer
        3. Provides actionable medical recommendations
        4. Assesses confidence in the solution
        
        Args:
            state (AgentState): Current system state with all research findings
            
        Returns:
            AgentState: Updated state with final solution and confidence score
        """
        print("💡 Solution Agent: Synthesizing patient data and research into final solution...")
        
        # Prepare patient analysis summary
        patient_analysis_summary = ""
        if state.get("patient_analysis"):
            patient_analysis_summary = "\n".join(state["patient_analysis"])
        elif state.get("patient_findings"):
            # Fallback to patient findings if analysis not available
            patient_summaries = []
            for finding in state["patient_findings"]:
                patient_summaries.append(f"Query: {finding['query']}\nAnswer: {finding['answer']}")
            patient_analysis_summary = "\n\n".join(patient_summaries)
        else:
            patient_analysis_summary = "No patient-specific information available."
        
        # Prepare library research summary
        library_research_summary = ""
        if state.get("extracted_information"):
            library_research_summary = "\n".join(state["extracted_information"])
            
            # Include detailed findings if available from enhanced library agent
            if state.get("detailed_findings"):
                library_research_summary += f"\n\nDETAILED SEARCH FINDINGS:\n"
                for i, finding in enumerate(state["detailed_findings"][:5], 1):  # Include top 5 findings
                    library_research_summary += f"{i}. Search term: '{finding['search_term']}' | "
                    library_research_summary += f"Source: {finding['source']} | "
                    library_research_summary += f"Relevance: {finding['relevance_score']:.3f}\n"
                    library_research_summary += f"   Content: {finding['content'][:200]}...\n\n"
        else:
            library_research_summary = "No research literature information available."
        
        # Prepare comprehensive prompt with all collected information
        prompt = SOLUTION_AGENT_PROMPT.format(
            user_query=state["user_query"],
            patient_id=state.get("patient_id", "Not specified"),
            consulting_analysis=state["consulting_analysis"],
            patient_analysis=patient_analysis_summary,
            extracted_information=library_research_summary,
            source_references=", ".join(state.get("source_references", []))
        )
        
        # Get the final synthesized response from LLM
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        response_text = response.content
        
        # Extract confidence score from the response
        confidence_match = re.search(r'CONFIDENCE ASSESSMENT:\s*.*?(\d+)', response_text)
        if confidence_match:
            # Convert from 1-10 scale to 0-1 scale
            confidence_score = float(confidence_match.group(1)) / 10
        else:
            # Calculate confidence based on available data
            has_patient_data = bool(state.get("patient_analysis") or state.get("patient_findings"))
            has_research_data = bool(state.get("extracted_information"))
            
            if has_patient_data and has_research_data:
                confidence_score = 0.9  # High confidence with both sources
            elif has_patient_data or has_research_data:
                confidence_score = 0.7  # Medium confidence with one source
            else:
                confidence_score = 0.5  # Low confidence with limited data
        
        # Update state with final results
        state["final_solution"] = response_text
        state["confidence_score"] = confidence_score
        
        # Add completion tracking message
        state["messages"].append(f"Solution Agent completed synthesis at {datetime.now()}")
        
        return state
    
    def process_query(self, user_query: str, patient_id: str = None, pdf_documents: List[str] = None) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system with both patient data and PDF processing.
        
        This is the main entry point that orchestrates the entire pipeline:
        1. Processes any provided PDF documents into vector embeddings
        2. Initializes the shared state with patient context
        3. Routes query through appropriate agents (patient, library, or both)
        4. Returns the complete results with integrated analysis
        
        Args:
            user_query (str): The user's question or request
            patient_id (str): Patient ID if query is patient-specific (optional)
            pdf_documents (List[str]): List of PDF file paths to search through (optional)
            
        Returns:
            Dict[str, Any]: Complete results from all agents including final solution
        """
        print(f"🚀 Starting integrated multi-agent processing for query: '{user_query}'")
        if patient_id:
            print(f"👤 Patient context: {patient_id}")
        print("=" * 80)
        
        # Process PDF documents if provided
        if pdf_documents:
            print(f"📄 Processing {len(pdf_documents)} PDF documents for semantic search...")
            self.enhanced_library_agent.process_documents(pdf_documents)
            print("✅ PDF processing completed!")
        else:
            print("📄 No PDF documents provided - using any previously processed documents")
        
        # Initialize the shared state that will flow through all agents
        initial_state = AgentState(
            user_query=user_query,
            patient_id=patient_id or "",
            consulting_analysis="",
            search_strategy=[],
            patient_queries=[],
            relevant_documents=[],
            extracted_information=[],
            source_references=[],
            detailed_findings=[],
            patient_data={},
            patient_analysis=[],
            patient_findings=[],
            preliminary_solution="",
            final_solution="",
            confidence_score=0.0,
            messages=[]
        )
        
        # Process the query through the entire agent pipeline
        # The graph handles the conditional routing: consulting → [patient + library] → solution
        final_state = self.graph.invoke(initial_state)
        
        print("=" * 80)
        print("✅ Integrated multi-agent processing completed!")
        
        # Return the final state which contains all agent outputs
        return final_state

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """
    Example usage of the integrated multi-agent system with patient data and PDF processing.
    
    This function demonstrates how to:
    1. Initialize the system with API credentials and patient data
    2. Process patient-specific queries with medical research
    3. Display comprehensive results including both patient data and research findings
    """
    # Initialize the system (requires OpenAI API key in environment)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create the integrated multi-agent system instance
    system = MultiAgentSystem(api_key, patients_json_path='./patients.json')
    
    # Example PDF documents to process (replace with actual PDF paths)
    pdf_documents = [
        "./ehab368.pdf"
    ]
    
    # Example 1: Patient-specific medical query
    print("\n" + "=" * 80)
    print("EXAMPLE 1: PATIENT-SPECIFIC MEDICAL QUERY")
    print("=" * 80)
    
    patient_query = "What are the cardiovascular risk factors for this patient based on their current medications and medical history?"
    patient_id = "p_001"  # Replace with actual patient ID
    
    results1 = system.process_query(
        user_query=patient_query,
        patient_id=patient_id,
        pdf_documents=pdf_documents
    )
    
    print(f"\nOriginal Query: {results1['user_query']}")
    print(f"Patient ID: {results1['patient_id']}")
    print(f"\nConsulting Analysis:\n{results1['consulting_analysis']}")
    if results1.get('patient_queries'):
        print(f"\nPatient Queries: {results1['patient_queries']}")
    if results1.get('search_strategy'):
        print(f"\nLibrary Search Strategy: {results1['search_strategy']}")
    print(f"\nFinal Solution:\n{results1['final_solution']}")
    print(f"\nConfidence Score: {results1['confidence_score']}")
    
    # Example 2: General medical research query (no patient context)
    print("\n" + "=" * 80)
    print("EXAMPLE 2: GENERAL MEDICAL RESEARCH QUERY")
    print("=" * 80)
    
    research_query = "What are the latest treatment protocols for acute heart failure management?"
    
    results2 = system.process_query(
        user_query=research_query,
        patient_id=None,
        pdf_documents=pdf_documents
    )
    
    print(f"\nOriginal Query: {results2['user_query']}")
    print(f"\nConsulting Analysis:\n{results2['consulting_analysis']}")
    if results2.get('search_strategy'):
        print(f"\nLibrary Search Strategy: {results2['search_strategy']}")
    print(f"\nFinal Solution:\n{results2['final_solution']}")
    print(f"\nConfidence Score: {results2['confidence_score']}")
    
    # Example 3: Mixed query requiring both patient data and research
    print("\n" + "=" * 80)
    print("EXAMPLE 3: INTEGRATED PATIENT AND RESEARCH QUERY")
    print("=" * 80)
    
    mixed_query = "Based on this patient's current lung cancer diagnosis and medications, what are the evidence-based treatment recommendations and potential drug interactions I should be aware of?"
    
    results3 = system.process_query(
        user_query=mixed_query,
        patient_id=patient_id,
        pdf_documents=pdf_documents
    )
    
    print(f"\nOriginal Query: {results3['user_query']}")
    print(f"Patient ID: {results3['patient_id']}")
    print(f"\nConsulting Analysis:\n{results3['consulting_analysis']}")
    if results3.get('patient_queries'):
        print(f"\nPatient Queries: {results3['patient_queries']}")
    if results3.get('search_strategy'):
        print(f"\nLibrary Search Strategy: {results3['search_strategy']}")
    
    # Display patient findings if available
    if results3.get('patient_findings'):
        print(f"\nPatient Findings Summary:")
        for i, finding in enumerate(results3['patient_findings'], 1):
            print(f"  {i}. {finding['query']}: {finding['answer'][:100]}...")
    
    # Display research findings if available
    if results3.get('detailed_findings'):
        print(f"\nResearch Findings Summary:")
        for i, finding in enumerate(results3['detailed_findings'][:3], 1):
            print(f"  {i}. Source: {finding['source']} (Relevance: {finding['relevance_score']:.3f})")
            print(f"     Content: {finding['content'][:100]}...")
    
    print(f"\nFinal Integrated Solution:\n{results3['final_solution']}")
    print(f"\nConfidence Score: {results3['confidence_score']}")
    
    print("\n" + "=" * 80)
    print("✅ Integrated multi-agent system demonstration completed!")

# Entry point for running the script directly
if __name__ == "__main__":
    main() 