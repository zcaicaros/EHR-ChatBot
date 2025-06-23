"""
Multi-Agent System using LangGraph
==================================

This system consists of three specialized agents:
1. Consulting Agent: Analyzes user queries and plans the information gathering strategy
2. Library Agent: Searches through PDF documents to find relevant information
3. Solution Agent: Synthesizes information to provide comprehensive solutions

Author: AI Assistant
"""

import os
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import PyPDF2
import re
from datetime import datetime

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    Represents the state that flows through our multi-agent system.
    Each agent can read from and write to this shared state.
    """
    # User's original query
    user_query: str
    
    # Consulting agent's analysis and strategy
    consulting_analysis: str
    search_strategy: List[str]
    information_requirements: List[str]
    
    # Library agent's findings
    relevant_documents: List[Dict[str, Any]]
    extracted_information: List[str]
    source_references: List[str]
    
    # Solution agent's work
    preliminary_solution: str
    final_solution: str
    confidence_score: float
    
    # System messages for debugging/tracking
    messages: List[str]

# ============================================================================
# AGENT PROMPTS
# ============================================================================

CONSULTING_AGENT_PROMPT = """
You are a Senior Consulting Agent with expertise in problem analysis and strategic thinking.

Your role is to:
1. DEEPLY ANALYZE the user's query to understand the underlying needs and context
2. IDENTIFY what specific information would be most valuable to answer their question
3. DEVELOP a strategic approach for information gathering
4. ANTICIPATE potential follow-up questions or related concerns

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

INFORMATION REQUIREMENTS:
[List 3-5 specific types of information needed to provide a comprehensive answer]

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
Information Requirements: {information_requirements}

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
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def search_text_for_keywords(text: str, keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Search for keywords in text and return relevant excerpts.
    
    Args:
        text (str): Text to search through
        keywords (List[str]): Keywords to search for
        
    Returns:
        List[Dict[str, Any]]: List of relevant excerpts with context
    """
    findings = []
    text_lower = text.lower()
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                
                # Extract context around the keyword (200 chars before and after)
                context_start = max(0, pos - 200)
                context_end = min(len(text), pos + len(keyword) + 200)
                context = text[context_start:context_end].strip()
                
                findings.append({
                    'keyword': keyword,
                    'context': context,
                    'position': pos,
                    'relevance_score': len(keyword)  # Simple scoring
                })
                
                start = pos + 1
    
    # Sort by relevance score (descending)
    findings.sort(key=lambda x: x['relevance_score'], reverse=True)
    return findings

# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class MultiAgentSystem:
    """
    Main class that orchestrates the multi-agent system using LangGraph.
    """
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4"):
        """
        Initialize the multi-agent system.
        
        Args:
            openai_api_key (str): OpenAI API key
            model_name (str): OpenAI model to use (default: gpt-4)
        """
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1  # Low temperature for consistent, focused responses
        )
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow that defines how agents interact.
        
        Returns:
            StateGraph: Configured LangGraph workflow
        """
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("consulting_agent", self.consulting_agent)
        workflow.add_node("library_agent", self.library_agent)
        workflow.add_node("solution_agent", self.solution_agent)
        
        # Define the workflow edges (how agents connect)
        workflow.add_edge("consulting_agent", "library_agent")
        workflow.add_edge("library_agent", "solution_agent")
        workflow.add_edge("solution_agent", END)
        
        # Set the entry point
        workflow.set_entry_point("consulting_agent")
        
        return workflow.compile()
    
    def consulting_agent(self, state: AgentState) -> AgentState:
        """
        Consulting Agent: Analyzes user query and develops information gathering strategy.
        
        Args:
            state (AgentState): Current system state
            
        Returns:
            AgentState: Updated state with consulting analysis
        """
        print("🧠 Consulting Agent: Analyzing query and developing strategy...")
        
        # Prepare the prompt
        prompt = CONSULTING_AGENT_PROMPT.format(
            user_query=state["user_query"]
        )
        
        # Get response from LLM
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Parse the response to extract structured information
        response_text = response.content
        
        # Extract analysis
        analysis_match = re.search(r'ANALYSIS:\s*(.*?)\s*SEARCH STRATEGY:', response_text, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else response_text
        
        # Extract search strategy
        strategy_match = re.search(r'SEARCH STRATEGY:\s*(.*?)\s*INFORMATION REQUIREMENTS:', response_text, re.DOTALL)
        search_strategy = []
        if strategy_match:
            strategy_text = strategy_match.group(1).strip()
            search_strategy = [item.strip() for item in strategy_text.split('\n') if item.strip()]
        
        # Extract information requirements
        requirements_match = re.search(r'INFORMATION REQUIREMENTS:\s*(.*?)$', response_text, re.DOTALL)
        information_requirements = []
        if requirements_match:
            requirements_text = requirements_match.group(1).strip()
            information_requirements = [item.strip() for item in requirements_text.split('\n') if item.strip()]
        
        # Update state
        state["consulting_analysis"] = analysis
        state["search_strategy"] = search_strategy
        state["information_requirements"] = information_requirements
        state["messages"].append(f"Consulting Agent completed analysis at {datetime.now()}")
        
        return state
    
    def library_agent(self, state: AgentState) -> AgentState:
        """
        Library Agent: Searches through documents and extracts relevant information.
        
        Args:
            state (AgentState): Current system state
            
        Returns:
            AgentState: Updated state with research findings
        """
        print("📚 Library Agent: Searching documents for relevant information...")
        
        # For this example, we'll simulate document search
        # In a real implementation, you would load actual PDF documents
        available_documents = ["sample_document.pdf"]  # Placeholder
        
        # Prepare the prompt
        prompt = LIBRARY_AGENT_PROMPT.format(
            search_strategy=state["search_strategy"],
            information_requirements=state["information_requirements"],
            available_documents=available_documents
        )
        
        # Get response from LLM
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Parse the response to extract findings
        response_text = response.content
        
        # Extract findings (simplified parsing)
        findings = re.findall(r'FINDING #\d+:(.*?)(?=FINDING #\d+:|$)', response_text, re.DOTALL)
        extracted_information = [finding.strip() for finding in findings]
        
        # Update state
        state["relevant_documents"] = [{"name": doc, "status": "processed"} for doc in available_documents]
        state["extracted_information"] = extracted_information if extracted_information else [response_text]
        state["source_references"] = available_documents
        state["messages"].append(f"Library Agent completed research at {datetime.now()}")
        
        return state
    
    def solution_agent(self, state: AgentState) -> AgentState:
        """
        Solution Agent: Synthesizes information into a comprehensive solution.
        
        Args:
            state (AgentState): Current system state
            
        Returns:
            AgentState: Updated state with final solution
        """
        print("💡 Solution Agent: Synthesizing information into final solution...")
        
        # Prepare the prompt
        prompt = SOLUTION_AGENT_PROMPT.format(
            user_query=state["user_query"],
            consulting_analysis=state["consulting_analysis"],
            extracted_information="\n".join(state["extracted_information"]),
            source_references=", ".join(state["source_references"])
        )
        
        # Get response from LLM
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        response_text = response.content
        
        # Extract confidence score (simplified)
        confidence_match = re.search(r'CONFIDENCE ASSESSMENT:\s*.*?(\d+)', response_text)
        confidence_score = float(confidence_match.group(1)) / 10 if confidence_match else 0.8
        
        # Update state
        state["final_solution"] = response_text
        state["confidence_score"] = confidence_score
        state["messages"].append(f"Solution Agent completed synthesis at {datetime.now()}")
        
        return state
    
    def process_query(self, user_query: str, pdf_documents: List[str] = None) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system.
        
        Args:
            user_query (str): The user's question or request
            pdf_documents (List[str]): List of PDF file paths to search through
            
        Returns:
            Dict[str, Any]: Complete results from all agents
        """
        print(f"🚀 Starting multi-agent processing for query: '{user_query}'")
        print("=" * 80)
        
        # Initialize state
        initial_state = AgentState(
            user_query=user_query,
            consulting_analysis="",
            search_strategy=[],
            information_requirements=[],
            relevant_documents=[],
            extracted_information=[],
            source_references=[],
            preliminary_solution="",
            final_solution="",
            confidence_score=0.0,
            messages=[]
        )
        
        # Process through the graph
        final_state = self.graph.invoke(initial_state)
        
        print("=" * 80)
        print("✅ Multi-agent processing completed!")
        
        return final_state

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """
    Example usage of the multi-agent system.
    """
    # Initialize the system (you need to provide your OpenAI API key)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    system = MultiAgentSystem(api_key)
    
    # Example query
    user_query = "What are the best practices for implementing microservices architecture in a large enterprise?"
    
    # Process the query
    results = system.process_query(user_query)
    
    # Display results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nOriginal Query: {results['user_query']}")
    print(f"\nConsulting Analysis:\n{results['consulting_analysis']}")
    print(f"\nSearch Strategy: {results['search_strategy']}")
    print(f"\nFinal Solution:\n{results['final_solution']}")
    print(f"\nConfidence Score: {results['confidence_score']}")

if __name__ == "__main__":
    main() 