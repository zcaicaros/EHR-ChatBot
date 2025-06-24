#!/usr/bin/env python3
"""
Multi-Agent Medical Consultation System - Example Usage
======================================================

This script demonstrates how to use the integrated multi-agent medical consultation system
for different types of medical queries. The system combines patient data analysis with
medical literature research to provide comprehensive clinical recommendations.

Prerequisites:
1. Set your OpenAI API key: export OPENAI_API_KEY='your-api-key-here'
2. Ensure you have patients.json with patient medical records
3. Optionally provide PDF medical literature documents

Usage Examples:
- Patient-specific queries: Analyze individual patient data with literature support
- General medical queries: Research-based medical recommendations
- Integrated queries: Combine patient data with evidence-based medicine

Author: Medical AI Team
"""

import os
from multiagent_system import MultiAgentSystem

def main():
    """
    Demonstrate the multi-agent medical consultation system capabilities.
    """
    print("🏥 Multi-Agent Medical Consultation System - Example Usage")
    print("=" * 70)
    
    # Check for required API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: Please set your OPENAI_API_KEY environment variable")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'")
        print("   You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    # Initialize the medical consultation system
    try:
        print("🔧 Initializing Multi-Agent Medical Consultation System...")
        system = MultiAgentSystem(
            openai_api_key=api_key,
            patients_json_path='./patients.json',  # Update path as needed
            model_name="gpt-4"  # Use gpt-4 for best medical analysis
        )
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Optional: Medical literature documents (PDFs)
    medical_literature = [
        # Add paths to your medical PDF documents here
        # "./medical_guidelines.pdf",
        # "./clinical_protocols.pdf",
        # "./research_papers.pdf"
        "./ehab368.pdf"
    ]
    
    print("\n" + "=" * 70)
    print("📋 EXAMPLE 1: PATIENT-SPECIFIC MEDICAL CONSULTATION")
    print("=" * 70)
    
    # Example 1: Patient-specific query
    patient_query = """
    What are the current cardiovascular risk factors for patient p_001 
    based on their medications, medical history, and recent lab results?
    """
    
    try:
        print(f"🔍 Processing Query: {patient_query.strip()}")
        print("⏳ Running multi-agent analysis...")
        
        result = system.process_medical_query(
            user_query=patient_query,
            patient_id="p_001",  # Replace with actual patient ID from your data
            pdf_documents=medical_literature
        )
        
        # Display results
        print(f"\n📊 CONSULTATION RESULTS:")
        print(f"Clinical Confidence: {result['confidence_score']:.1%}")
        print(f"\n🩺 Clinical Analysis:")
        print(result['consulting_analysis'][:300] + "..." if len(result['consulting_analysis']) > 300 else result['consulting_analysis'])
        
        if result.get('patient_queries'):
            print(f"\n👤 Patient Data Analyzed: {len(result['patient_queries'])} queries")
        
        if result.get('search_strategy'):
            print(f"📚 Literature Searched: {len(result['search_strategy'])} medical topics")
        
        print(f"\n🏥 MEDICAL RECOMMENDATIONS:")
        # Show first part of recommendations
        recommendations = result['final_solution']
        # if len(recommendations) > 500:
        #     print(recommendations[:500] + "\n... [truncated for display]")
        # else:
            # print(recommendations)
        print(recommendations)
            
    except Exception as e:
        print(f"❌ Error in patient consultation: {e}")
    
    # print("\n" + "=" * 70)
    # print("📚 EXAMPLE 2: GENERAL MEDICAL RESEARCH QUERY")
    # print("=" * 70)
    
    # # Example 2: General medical research
    # research_query = """
    # What are the latest evidence-based guidelines for managing 
    # acute heart failure in elderly patients?
    # """
    
    # try:
    #     print(f"🔍 Processing Query: {research_query.strip()}")
    #     print("⏳ Running literature analysis...")
        
    #     result = system.process_medical_query(
    #         user_query=research_query,
    #         patient_id=None,  # No specific patient
    #         pdf_documents=medical_literature
    #     )
        
    #     # Display results
    #     print(f"\n📊 RESEARCH RESULTS:")
    #     print(f"Clinical Confidence: {result['confidence_score']:.1%}")
        
    #     if result.get('search_strategy'):
    #         print(f"📚 Literature Topics Researched: {result['search_strategy']}")
        
    #     print(f"\n🏥 EVIDENCE-BASED RECOMMENDATIONS:")
    #     recommendations = result['final_solution']
    #     if len(recommendations) > 500:
    #         print(recommendations[:500] + "\n... [truncated for display]")
    #     else:
    #         print(recommendations)
            
    # except Exception as e:
    #     print(f"❌ Error in research query: {e}")
    
    print("\n" + "=" * 70)
    print("💡 USAGE TIPS")
    print("=" * 70)
    print("""
    📋 For Patient-Specific Queries:
    - Use patient IDs that exist in your patients.json file
    - Ask about medications, lab results, medical history, risk factors
    - The system will extract only relevant patient data to optimize performance
    
    📚 For Literature Research:
    - Provide PDF documents with medical literature, guidelines, or protocols
    - Ask about treatment protocols, evidence-based recommendations, drug interactions
    - The system uses semantic search to find relevant medical information
    
    🔧 System Features:
    - 72% reduction in context usage through intelligent field extraction
    - Natural language processing for complex medical queries
    - Integration of patient data with evidence-based medicine
    - Confidence scoring for clinical recommendations
    - Source attribution for medical verification
    
    ⚠️  Important Notes:
    - This system is for educational and research purposes
    - Always consult qualified healthcare professionals for medical decisions
    - Verify all recommendations with current medical guidelines
    - Maintain patient privacy and follow HIPAA guidelines
    """)
    
    print("\n✅ Example demonstration completed!")
    print("🏥 The Multi-Agent Medical Consultation System is ready for use.")

if __name__ == "__main__":
    main() 