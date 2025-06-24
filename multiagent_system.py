"""
Multi-Agent Medical Consultation System
======================================

An intelligent multi-agent system that combines patient data analysis with medical literature research
to provide comprehensive medical insights and recommendations.

System Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Consulting      │    │ Patient Agent    │    │ Library Agent    │    │ Solution Agent  │
│ Agent           │───▶│ (Patient Data)   │───▶│ (Medical Lit)    │───▶│ (Synthesis)     │
│ (Analysis &     │    │                  │    │                  │    │                 │
│ Strategy)       │    └──────────────────┘    └──────────────────┘    └─────────────────┘
└─────────────────┘              │                       │                       │
                                 │                       │                       │
                            Extracts patient        Searches medical         Combines both sources
                            medical records         literature & PDFs        for final recommendations

Key Features:
- Natural language query processing for medical questions
- Patient-specific data extraction and analysis  
- Medical literature search through PDF documents
- Intelligent routing based on query requirements
- Comprehensive medical recommendations with confidence scoring

Author: Medical AI Team
Version: 2.0 (Integrated Patient + Library System)
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
# SHARED STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    Shared state that flows through the multi-agent system.
    
    This acts as a collaborative workspace where each agent contributes specialized knowledge:
    - Consulting Agent: Analyzes queries and plans information gathering strategy
    - Patient Agent: Extracts and analyzes patient-specific medical data
    - Library Agent: Searches medical literature and research documents
    - Solution Agent: Synthesizes all information into comprehensive recommendations
    
    The state accumulates information as it flows through the pipeline, ensuring each
    agent has access to all previous work for context and decision-making.
    """
    
    # ---- Core Query Information ----
    user_query: str                          # Original question from user
    patient_id: str                          # Patient ID if query is patient-specific
    
    # ---- Consulting Agent Results ----
    # Strategic analysis and planning phase
    consulting_analysis: str                 # Deep analysis of user needs and context
    search_strategy: List[str]               # Specific search terms for medical literature
    patient_queries: List[str]               # Specific questions about patient data
    
    # ---- Patient Agent Results ----  
    # Patient-specific medical data and analysis
    patient_data: Dict[str, Any]             # Raw patient data extracted from records
    patient_analysis: List[str]              # Medical analysis and interpretation
    patient_findings: List[Dict[str, Any]]   # Detailed findings with confidence scores
    
    # ---- Library Agent Results ----
    # Medical literature research and evidence
    relevant_documents: List[Dict[str, Any]] # Documents that were searched and processed
    extracted_information: List[str]         # Key information from medical literature
    source_references: List[str]             # Source citations for verification
    detailed_findings: List[Dict[str, Any]]  # Detailed search results with relevance scores
    
    # ---- Solution Agent Results ----
    # Final synthesis and recommendations
    preliminary_solution: str                # Initial draft (currently unused)
    final_solution: str                      # Complete medical recommendations
    confidence_score: float                  # System confidence in recommendations (0-1)
    
    # ---- System Tracking ----
    # Debugging and process monitoring
    messages: List[str]                      # Processing timestamps and status messages

# ============================================================================
# AGENT PROMPTS - MEDICAL CONSULTATION TEMPLATES
# ============================================================================

CONSULTING_AGENT_PROMPT = """
You are a Senior Medical Consulting Agent with expertise in clinical problem analysis and research strategy development.

ROLE OVERVIEW:
Your primary responsibility is to analyze complex medical queries and develop targeted information gathering strategies. 
You determine what patient-specific data is needed and what medical literature should be researched to provide 
comprehensive, evidence-based medical recommendations.

ANALYSIS FRAMEWORK:
1. CLINICAL CONTEXT: Understand the medical context and urgency of the query
2. INFORMATION NEEDS: Identify what patient data and research evidence is required
3. SEARCH STRATEGY: Develop specific, targeted search approaches
4. PATIENT FOCUS: Determine if and how patient-specific data should be analyzed

GUIDELINES:
- Think like a senior clinician approaching a complex case
- Consider both patient-specific factors and general medical evidence
- Prioritize information that directly impacts clinical decision-making
- Ensure comprehensive coverage while avoiding information overload
- Be specific in your recommendations for data extraction and literature search

USER QUERY: {user_query}
PATIENT ID: {patient_id}

REQUIRED OUTPUT FORMAT:

ANALYSIS:
[Provide thorough clinical analysis including:
- Medical context and significance of the query
- Key clinical decision points that need to be addressed
- Underlying medical concepts and relationships
- Potential complications or considerations]

PATIENT QUERIES:
[List 2-4 specific, focused questions about patient data needed, such as:
- Current medications and potential interactions
- Relevant diagnostic results and trends
- Medical history pertinent to the query
- Current treatment status and outcomes
Write "None" if no patient data is needed]

LIBRARY SEARCH STRATEGY:
[List 2-4 specific medical research topics or evidence areas, such as:
- Treatment guidelines and protocols
- Drug interaction studies
- Clinical outcomes research
- Diagnostic criteria and recommendations
Write "None" if no literature research is needed]

Focus on clinical relevance and actionable medical insights.
"""

PATIENT_AGENT_PROMPT = """
You are a Patient Data Analysis Agent specializing in medical record extraction and clinical data interpretation.

ROLE OVERVIEW:
Your expertise lies in efficiently extracting relevant patient information and providing clinical context 
for medical decision-making. You analyze patient records to identify patterns, trends, and clinically 
significant findings that directly address the consulting agent's queries.

CLINICAL ANALYSIS FRAMEWORK:
1. DATA EXTRACTION: Retrieve specific medical information requested
2. CLINICAL INTERPRETATION: Provide medical context for findings
3. TREND ANALYSIS: Identify patterns in patient data over time
4. RISK ASSESSMENT: Highlight clinically significant findings
5. CARE CONTINUITY: Consider how findings relate to ongoing care

GUIDELINES:
- Focus on clinically relevant data that impacts patient care
- Provide context for medical values, dates, and trends
- Highlight abnormal or concerning findings
- Maintain patient privacy while ensuring comprehensive analysis
- Include relevant medical codes and references for verification

PATIENT QUERIES TO ADDRESS: {patient_queries}
PATIENT ID: {patient_id}

REQUIRED OUTPUT FORMAT:

For each query, provide structured medical analysis:

PATIENT FINDING #[X]:
Query: [Restate the specific patient question being addressed]
Clinical Data: [Relevant patient information extracted from records]
Medical Interpretation: [Clinical significance and context]
Trends/Patterns: [Any relevant patterns or changes over time]
Clinical Significance: [How this impacts patient care and decision-making]

Emphasize accuracy, clinical relevance, and actionable medical insights.
"""

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

SOLUTION_AGENT_PROMPT = """
You are a Medical Solution Synthesis Agent specializing in integrating patient data with evidence-based medicine 
to provide comprehensive clinical recommendations.

ROLE OVERVIEW:
Your expertise is in combining patient-specific medical information with current medical literature to create 
actionable, evidence-based clinical recommendations. You think like a senior clinician making informed 
decisions based on both individual patient factors and established medical evidence.

CLINICAL SYNTHESIS FRAMEWORK:
1. PATIENT-SPECIFIC ANALYSIS: Consider individual patient factors and medical history
2. EVIDENCE INTEGRATION: Apply relevant medical literature to the specific case
3. RISK-BENEFIT ASSESSMENT: Evaluate potential outcomes and considerations
4. CLINICAL RECOMMENDATIONS: Provide specific, actionable medical guidance
5. SAFETY CONSIDERATIONS: Highlight important precautions and monitoring needs

GUIDELINES:
- Start with a clear, direct answer to the clinical question
- Integrate patient data with evidence-based medical recommendations
- Provide specific, actionable clinical guidance
- Address potential risks, contraindications, and monitoring requirements
- Acknowledge limitations and areas where additional information may be needed
- Maintain professional medical standards and ethical considerations

USER QUERY: {user_query}
PATIENT ID: {patient_id}
CONSULTING ANALYSIS: {consulting_analysis}

PATIENT-SPECIFIC INFORMATION:
{patient_analysis}

MEDICAL LITERATURE EVIDENCE:
{extracted_information}

SOURCE REFERENCES: {source_references}

REQUIRED OUTPUT FORMAT:

CLINICAL ANSWER:
[Direct, clear response to the medical query with specific recommendations]

PATIENT-SPECIFIC CONSIDERATIONS:
[Key patient factors that influence clinical decision-making]

EVIDENCE-BASED SUPPORT:
[Medical literature findings that support recommendations with source citations]

INTEGRATED CLINICAL ASSESSMENT:
[Comprehensive analysis combining patient data with medical evidence]

CLINICAL RECOMMENDATIONS:
[Specific, actionable medical recommendations including:]
- Treatment approaches or modifications
- Monitoring requirements
- Patient education needs
- Follow-up considerations

SAFETY AND MONITORING:
[Important precautions, contraindications, and monitoring parameters]

CLINICAL LIMITATIONS:
[Areas where additional information, testing, or specialist consultation may be needed]

CONFIDENCE ASSESSMENT:
[Rate confidence from 1-10 with clinical justification]

Prioritize patient safety and evidence-based clinical practice in all recommendations.
"""

# ============================================================================
# MULTI-AGENT MEDICAL CONSULTATION SYSTEM
# ============================================================================

class MultiAgentSystem:
    """
    Integrated Multi-Agent Medical Consultation System
    
    This system orchestrates four specialized AI agents to provide comprehensive medical insights:
    
    1. **Consulting Agent**: Acts like a senior physician analyzing the medical query
       - Understands clinical context and determines information needs
       - Plans targeted data extraction and literature search strategies
       
    2. **Patient Agent**: Functions as a medical records specialist
       - Extracts relevant patient data from medical records
       - Provides clinical interpretation and identifies significant patterns
       
    3. **Library Agent**: Operates as a medical research librarian
       - Searches medical literature and clinical guidelines
       - Evaluates evidence quality and clinical applicability
       
    4. **Solution Agent**: Works like a consulting physician
       - Synthesizes patient data with medical evidence
       - Provides comprehensive, evidence-based clinical recommendations
    
    WORKFLOW:
    User Query → Consulting Agent → Patient Agent → Library Agent → Solution Agent → Final Recommendations
    
    Each agent builds upon the work of previous agents, creating a comprehensive
    medical consultation that considers both patient-specific factors and current medical evidence.
    """
    
    def __init__(self, openai_api_key: str, patients_json_path: str = './patients.json', model_name: str = "gpt-4"):
        """
        Initialize the multi-agent medical consultation system.
        
        Args:
            openai_api_key (str): OpenAI API key for accessing GPT models
            patients_json_path (str): Path to patient data JSON file
            model_name (str): OpenAI model for medical analysis (default: gpt-4 for clinical accuracy)
        """
        # Initialize the primary language model with settings optimized for medical consultation
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1  # Low temperature for consistent, reliable medical analysis
        )
        
        # Initialize specialized agents with medical capabilities
        self.enhanced_library_agent = EnhancedLibraryAgent(openai_api_key)
        self.patient_query_agent = PatientQueryAgent(
            json_file_path=patients_json_path,
            openai_api_key=openai_api_key,
            model_name=model_name
        )
        
        # Build the medical consultation workflow
        self.graph = self._build_medical_workflow()
        
    def _build_medical_workflow(self) -> StateGraph:
        """
        Build the medical consultation workflow using LangGraph.
        
        Creates a sequential workflow that mirrors clinical consultation:
        Consulting Analysis → Patient Data Review → Literature Research → Clinical Recommendations
        
        This sequential approach ensures:
        - No data conflicts between parallel agents
        - Each agent has complete context from previous steps
        - Information builds progressively toward comprehensive recommendations
        
        Returns:
            StateGraph: Configured medical consultation workflow
        """
        # Create the workflow graph for medical consultation
        workflow = StateGraph(AgentState)
        
        # Add each specialized medical agent
        workflow.add_node("consulting_agent", self.consulting_agent)
        workflow.add_node("patient_agent", self.patient_agent)
        workflow.add_node("library_agent", self.library_agent)
        workflow.add_node("solution_agent", self.solution_agent)
        
        # Define intelligent routing based on medical consultation needs
        def route_after_consulting(state: AgentState):
            """Route to patient data analysis if patient-specific queries exist, otherwise to literature search."""
            patient_queries = state.get("patient_queries", [])
            search_strategy = state.get("search_strategy", [])
            
            # Priority: Patient data first (clinical information), then literature research
            if patient_queries and any(q.lower() != "none" for q in patient_queries):
                return "patient_agent"
            elif search_strategy and any(s.lower() != "none" for s in search_strategy):
                return "library_agent"
            else:
                return "solution_agent"
        
        def route_after_patient(state: AgentState):
            """Route to literature search if needed, otherwise proceed to clinical recommendations."""
            search_strategy = state.get("search_strategy", [])
            
            if search_strategy and any(s.lower() != "none" for s in search_strategy):
                return "library_agent"
            else:
                return "solution_agent"
        
        # Configure the medical consultation workflow routing
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
        
        # Final routing to clinical recommendations
        workflow.add_edge("library_agent", "solution_agent")
        workflow.add_edge("solution_agent", END)
        
        # Start with clinical analysis
        workflow.set_entry_point("consulting_agent")
        
        return workflow.compile()
    
    def consulting_agent(self, state: AgentState) -> AgentState:
        """
        Consulting Agent: Senior Medical Analyst
        
        Analyzes medical queries like a senior physician would approach a complex case:
        - Understands clinical context and identifies key decision points
        - Determines what patient data is needed for clinical assessment
        - Plans literature search strategy for evidence-based recommendations
        - Routes query to appropriate specialized analysis
        
        This agent acts as the "attending physician" who guides the overall consultation approach.
        
        Args:
            state (AgentState): Current consultation state with user query
            
        Returns:
            AgentState: Updated state with clinical analysis and information gathering strategy
        """
        print("🩺 Consulting Agent: Analyzing medical query and developing clinical strategy...")
        
        # Prepare medical consultation prompt with clinical context
        prompt = CONSULTING_AGENT_PROMPT.format(
            user_query=state["user_query"],
            patient_id=state.get("patient_id", "Not provided")
        )
        
        # Get clinical analysis from the AI physician
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # Parse the structured medical analysis response
        # Extract clinical analysis
        analysis_match = re.search(r'ANALYSIS:\s*(.*?)\s*PATIENT QUERIES:', response_text, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else response_text
        
        # Extract patient-specific queries
        patient_queries_match = re.search(r'PATIENT QUERIES:\s*(.*?)\s*LIBRARY SEARCH STRATEGY:', response_text, re.DOTALL)
        patient_queries = []
        if patient_queries_match:
            queries_text = patient_queries_match.group(1).strip()
            if queries_text.lower() != "none":
                patient_queries = [item.strip() for item in queries_text.split('\n') if item.strip()]
        
        # Extract medical literature search strategy
        strategy_match = re.search(r'LIBRARY SEARCH STRATEGY:\s*(.*?)$', response_text, re.DOTALL)
        search_strategy = []
        if strategy_match:
            strategy_text = strategy_match.group(1).strip()
            if strategy_text.lower() != "none":
                search_strategy = [item.strip() for item in strategy_text.split('\n') if item.strip()]
        
        # Update consultation state with clinical strategy
        state["consulting_analysis"] = analysis
        state["patient_queries"] = patient_queries
        state["search_strategy"] = search_strategy
        
        # Track consultation progress
        state["messages"].append(f"Consulting Agent completed clinical analysis at {datetime.now()}")
        
        return state

    def patient_agent(self, state: AgentState) -> AgentState:
        """
        Patient Agent: Medical Records Specialist
        
        Extracts and analyzes patient-specific medical data like a clinical data analyst:
        - Processes specific patient queries identified by the consulting agent
        - Extracts relevant medical information from patient records
        - Provides clinical interpretation and identifies significant patterns
        - Maintains patient privacy while ensuring comprehensive analysis
        
        This agent functions as the "medical records specialist" providing patient-specific context.
        
        Args:
            state (AgentState): Current consultation state with patient queries
            
        Returns:
            AgentState: Updated state with patient data analysis and clinical findings
        """
        print("📋 Patient Agent: Analyzing patient medical records...")
        
        patient_queries = state.get("patient_queries", [])
        patient_id = state.get("patient_id")
        
        # Handle cases where no patient analysis is needed
        if not patient_queries or not patient_id:
            print("   No patient queries or patient ID provided, skipping patient analysis")
            state["patient_data"] = {}
            state["patient_analysis"] = []
            state["patient_findings"] = []
            state["messages"].append(f"Patient Agent: No patient analysis required at {datetime.now()}")
            return state
        
        # Process each patient query using the specialized patient data agent
        patient_analysis = []
        patient_findings = []
        all_patient_data = {}
        
        for i, query in enumerate(patient_queries):
            print(f"   Processing patient query {i+1}: {query}")
            
            try:
                # Extract patient data using the PatientQueryAgent
                result = self.patient_query_agent.process_query(query, patient_id)
                
                # Structure the clinical findings
                finding = {
                    "query": query,
                    "patient_id": patient_id,
                    "answer": result.get("answer", "No answer available"),
                    "query_type": result.get("query_type", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "relevant_data": result.get("relevant_data", {})
                }
                patient_findings.append(finding)
                
                # Consolidate patient data for comprehensive analysis
                if result.get("relevant_data"):
                    for data_type, data in result["relevant_data"].items():
                        if data_type not in all_patient_data:
                            all_patient_data[data_type] = data
                        elif isinstance(data, list) and isinstance(all_patient_data[data_type], list):
                            all_patient_data[data_type].extend(data)
                
                # Generate clinical analysis using medical prompt
                analysis_prompt = PATIENT_AGENT_PROMPT.format(
                    patient_queries=[query],
                    patient_id=patient_id
                )
                analysis_prompt += f"\n\nClinical Data Retrieved:\n{result.get('answer', 'No data')}"
                
                messages = [SystemMessage(content=analysis_prompt)]
                analysis_response = self.llm.invoke(messages)
                patient_analysis.append(analysis_response.content)
                
            except Exception as e:
                print(f"   Error processing patient query: {e}")
                patient_analysis.append(f"Error processing query '{query}': {str(e)}")
        
        # Update consultation state with patient findings
        state["patient_data"] = all_patient_data
        state["patient_analysis"] = patient_analysis
        state["patient_findings"] = patient_findings
        
        # Track patient analysis completion
        state["messages"].append(f"Patient Agent completed analysis of {len(patient_queries)} queries at {datetime.now()}")
        
        return state
    
    def library_agent(self, state: AgentState) -> AgentState:
        """
        Library Agent: Medical Research Librarian
        
        Conducts comprehensive medical literature search using advanced vector-based semantic search:
        - Uses the consulting agent's research strategy to guide document search
        - Performs semantic search through medical PDFs and literature
        - Evaluates evidence quality and clinical relevance
        - Provides proper source attribution for medical verification
        
        This agent functions as the "medical librarian" providing evidence-based research support.
        
        Args:
            state (AgentState): Current consultation state with search strategy
            
        Returns:
            AgentState: Updated state with medical literature findings and evidence
        """
        print("📚 Library Agent: Conducting medical literature search...")
        
        # Delegate to the enhanced library agent with real PDF processing
        # This provides sophisticated vector-based semantic search capabilities
        enhanced_state = self.enhanced_library_agent.enhanced_library_search(state)
        
        # Track literature search completion
        enhanced_state["messages"].append(f"Library Agent completed medical literature search at {datetime.now()}")
        
        return enhanced_state
    
    def solution_agent(self, state: AgentState) -> AgentState:
        """
        Solution Agent: Senior Consulting Physician
        
        Synthesizes all information into comprehensive medical recommendations like a senior physician:
        - Reviews all patient data and medical literature evidence
        - Integrates patient-specific factors with evidence-based medicine
        - Provides actionable clinical recommendations with confidence assessment
        - Addresses safety considerations and monitoring requirements
        
        This agent functions as the "attending physician" providing final clinical recommendations.
        
        Args:
            state (AgentState): Complete consultation state with all findings
            
        Returns:
            AgentState: Updated state with final medical recommendations and confidence assessment
        """
        print("🏥 Solution Agent: Synthesizing clinical recommendations...")
        
        # Prepare comprehensive patient analysis summary
        patient_analysis_summary = ""
        if state.get("patient_analysis"):
            patient_analysis_summary = "\n".join(state["patient_analysis"])
        elif state.get("patient_findings"):
            # Alternative: Use patient findings if detailed analysis not available
            patient_summaries = []
            for finding in state["patient_findings"]:
                patient_summaries.append(f"Query: {finding['query']}\nFindings: {finding['answer']}")
            patient_analysis_summary = "\n\n".join(patient_summaries)
        else:
            patient_analysis_summary = "No patient-specific information available for this consultation."
        
        # Prepare medical literature evidence summary
        library_research_summary = ""
        if state.get("extracted_information"):
            library_research_summary = "\n".join(state["extracted_information"])
            
            # Include detailed search findings if available for transparency
            if state.get("detailed_findings"):
                library_research_summary += f"\n\nDETAILED MEDICAL LITERATURE FINDINGS:\n"
                for i, finding in enumerate(state["detailed_findings"][:5], 1):  # Top 5 most relevant
                    library_research_summary += f"{i}. Search: '{finding['search_term']}' | "
                    library_research_summary += f"Source: {finding['source']} | "
                    library_research_summary += f"Relevance: {finding['relevance_score']:.3f}\n"
                    library_research_summary += f"   Evidence: {finding['content'][:200]}...\n\n"
        else:
            library_research_summary = "No relevant medical literature found for this consultation."
        
        # Create comprehensive medical consultation prompt
        prompt = SOLUTION_AGENT_PROMPT.format(
            user_query=state["user_query"],
            patient_id=state.get("patient_id", "Not specified"),
            consulting_analysis=state["consulting_analysis"],
            patient_analysis=patient_analysis_summary,
            extracted_information=library_research_summary,
            source_references=", ".join(state.get("source_references", []))
        )
        
        # Generate final medical recommendations
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # Calculate clinical confidence score
        confidence_match = re.search(r'CONFIDENCE ASSESSMENT:\s*.*?(\d+)', response_text)
        if confidence_match:
            # Convert from 1-10 clinical scale to 0-1 system scale
            confidence_score = float(confidence_match.group(1)) / 10
        else:
            # Calculate confidence based on available clinical information
            has_patient_data = bool(state.get("patient_analysis") or state.get("patient_findings"))
            has_research_data = bool(state.get("extracted_information"))
            
            if has_patient_data and has_research_data:
                confidence_score = 0.9  # High confidence: both patient data and evidence
            elif has_patient_data or has_research_data:
                confidence_score = 0.7  # Medium confidence: single source
            else:
                confidence_score = 0.5  # Low confidence: limited information
        
        # Finalize consultation state with medical recommendations
        state["final_solution"] = response_text
        state["confidence_score"] = confidence_score
        
        # Track consultation completion
        state["messages"].append(f"Solution Agent completed medical consultation at {datetime.now()}")
        
        return state
    
    def process_medical_query(self, user_query: str, patient_id: str = None, pdf_documents: List[str] = None) -> Dict[str, Any]:
        """
        Process a medical query through the complete multi-agent consultation system.
        
        This is the main entry point for medical consultations that:
        1. Processes medical literature (PDFs) for evidence-based research
        2. Analyzes patient-specific medical data if patient ID provided
        3. Routes the query through appropriate medical specialists (agents)
        4. Returns comprehensive medical recommendations with confidence assessment
        
        CLINICAL WORKFLOW:
        User Query → Clinical Analysis → Patient Data Review → Literature Search → Medical Recommendations
        
        Args:
            user_query (str): Medical question or consultation request
            patient_id (str, optional): Patient identifier for patient-specific queries
            pdf_documents (List[str], optional): Medical literature PDFs to search
            
        Returns:
            Dict[str, Any]: Complete medical consultation results including:
                - Clinical analysis and strategy
                - Patient-specific findings (if applicable)
                - Medical literature evidence (if applicable)
                - Comprehensive recommendations with confidence score
        """
        print(f"🏥 Starting Medical Consultation for: '{user_query}'")
        if patient_id:
            print(f"👤 Patient ID: {patient_id}")
        print("=" * 80)
        
        # Process medical literature if provided
        if pdf_documents:
            print(f"📚 Processing {len(pdf_documents)} medical documents for literature search...")
            self.enhanced_library_agent.process_documents(pdf_documents)
            print("✅ Medical literature processing completed!")
        else:
            print("📚 No medical documents provided - using previously processed literature")
        
        # Initialize medical consultation state
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
        
        # Execute the complete medical consultation workflow
        final_state = self.graph.invoke(initial_state)
        
        print("=" * 80)
        print("✅ Medical consultation completed successfully!")
        
        return final_state

# ============================================================================
# EXAMPLE MEDICAL CONSULTATION USAGE
# ============================================================================

def main():
    """
    Example usage of the Multi-Agent Medical Consultation System.
    
    Demonstrates how to:
    1. Initialize the system with medical credentials and patient data
    2. Process different types of medical queries (patient-specific and general)
    3. Integrate medical literature search with patient data analysis
    4. Display comprehensive medical consultation results
    """
    # Initialize system with OpenAI credentials
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create the medical consultation system
    system = MultiAgentSystem(api_key, patients_json_path='./patients.json')
    
    # Example medical literature documents
    medical_documents = [
        "./ehab368.pdf"  # Replace with actual medical literature paths
    ]
    
    # Example 1: Patient-specific medical consultation
    print("\n" + "=" * 80)
    print("EXAMPLE 1: PATIENT-SPECIFIC MEDICAL CONSULTATION")
    print("=" * 80)
    
    patient_query = "What are the cardiovascular risk factors for this patient based on their current medications and medical history?"
    patient_id = "p_001"  # Replace with actual patient ID from your data
    
    results1 = system.process_medical_query(
        user_query=patient_query,
        patient_id=patient_id,
        pdf_documents=medical_documents
    )
    
    # Display consultation results
    print(f"\n🔍 Medical Query: {results1['user_query']}")
    print(f"👤 Patient ID: {results1['patient_id']}")
    print(f"\n📋 Clinical Analysis:\n{results1['consulting_analysis']}")
    if results1.get('patient_queries'):
        print(f"\n🩺 Patient Data Queries: {results1['patient_queries']}")
    if results1.get('search_strategy'):
        print(f"\n📚 Literature Search Strategy: {results1['search_strategy']}")
    print(f"\n🏥 Medical Recommendations:\n{results1['final_solution']}")
    print(f"\n📊 Clinical Confidence: {results1['confidence_score']:.1%}")
    
    # Example 2: General medical research query
    print("\n" + "=" * 80)
    print("EXAMPLE 2: GENERAL MEDICAL RESEARCH CONSULTATION")
    print("=" * 80)
    
    research_query = "What are the latest evidence-based treatment protocols for acute heart failure management?"
    
    results2 = system.process_medical_query(
        user_query=research_query,
        patient_id=None,
        pdf_documents=medical_documents
    )
    
    print(f"\n🔍 Medical Query: {results2['user_query']}")
    print(f"\n📋 Clinical Analysis:\n{results2['consulting_analysis']}")
    if results2.get('search_strategy'):
        print(f"\n📚 Literature Search Strategy: {results2['search_strategy']}")
    print(f"\n🏥 Evidence-Based Recommendations:\n{results2['final_solution']}")
    print(f"\n📊 Clinical Confidence: {results2['confidence_score']:.1%}")
    
    print("\n" + "=" * 80)
    print("✅ Medical consultation system demonstration completed!")
    print("   Ready for clinical use with appropriate medical oversight.")

# For backwards compatibility - alias the main method
process_query = lambda self, *args, **kwargs: self.process_medical_query(*args, **kwargs)

if __name__ == "__main__":
    main() 