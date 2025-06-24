import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class PatientQueryAgent:
    """
    An intelligent patient query agent that can handle natural language questions
    about patients and return relevant information from their medical records.
    """
    
    def __init__(self, json_file_path: str, openai_api_key: str = None, model_name: str = "gpt-4"):
        """
        Initialize the agent with a JSON file containing patient records and AI capabilities.
        
        Args:
            json_file_path (str): Path to the patients.json file
            openai_api_key (str): OpenAI API key for AI capabilities
            model_name (str): OpenAI model to use for query processing
        """
        self.json_file_path = json_file_path
        self.patients_data = []
        self.load_patient_data()
        
        # Initialize AI model if API key is provided
        self.llm = None
        if openai_api_key:
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model=model_name,
                temperature=0
            )
    
    def load_patient_data(self):
        """Load and parse the patient data from the JSON file."""
        try:
            with open(self.json_file_path, 'r') as file:
                self.raw_data = json.load(file)
                self.patients_data = self.raw_data
                print(f"Successfully loaded data for {len(self.patients_data)} patients")
        except FileNotFoundError:
            print(f"Error: File {self.json_file_path} not found")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {self.json_file_path}")
        except Exception as e:
            print(f"Error loading patient data: {str(e)}")

    def process_query(self, query: str, patient_id: str = None) -> Dict[str, Any]:
        """
        Process a natural language query about patient(s) and return relevant information.
        
        Args:
            query (str): Natural language question about patient(s)
            patient_id (str, optional): Specific patient ID to query about
            
        Returns:
            Dict: Response containing answer, relevant data, and metadata
        """
        # If no AI model available, fall back to simple query processing
        if not self.llm:
            return self._process_simple_query(query, patient_id)
        
        # Parse the query to understand what information is being requested
        query_analysis = self._analyze_query(query, patient_id)
        
        # Extract relevant patient data based on the analysis
        relevant_data = self._extract_relevant_data(query_analysis, patient_id)
        
        # Generate a comprehensive response using the AI model
        response = self._generate_response(query, query_analysis, relevant_data)
        
        return {
            "query": query,
            "patient_id": patient_id,
            "answer": response,
            "relevant_data": relevant_data,
            "query_type": query_analysis.get("query_type"),
            "confidence": query_analysis.get("confidence", 0.8)
        }

    def _analyze_query(self, query: str, patient_id: str = None) -> Dict[str, Any]:
        """Analyze the user's query to understand what information they're seeking."""
        
        analysis_prompt = f"""
        Analyze the following patient-related query and determine EXACTLY which specific fields are needed.
        Be very specific to minimize data retrieval and avoid context length issues.
        
        Query: "{query}"
        Patient ID (if specified): {patient_id or "Not specified"}
        
        Available patient data fields:
        Demographics: name, gender, dob/birthDate, address, contact.phone, contact.email
        Encounters: status, class.display, type, period.start, period.end, reasonCode, diagnosis
        Diagnoses: code.code, code.display, effectiveDateTime, category
        Medications: genericName, brandName, dosage, startDate, endDate, frequency
        Procedures: code.display, performedDateTime, outcome, bodySite
        Labs: code.display, valueQuantity.value, valueQuantity.unit, effectiveDateTime, status
        Allergies: allergenname, reaction, severity, onset
        Visits: visitType, department, period.start, period.end, status
        
        Respond in JSON format with SPECIFIC fields only:
        {{
            "query_type": "demographics|medications|diagnoses|labs|procedures|allergies|encounters|visits|comprehensive|search",
            "specific_fields": ["exact.field.names", "to.extract"],
            "data_types": ["demographics", "medications", etc.],
            "scope": "single_patient|multiple_patients|search_patients",
            "search_criteria": {{"field": "value"}} if searching,
            "confidence": 0.0-1.0,
            "max_records_per_type": 5
        }}
        """
        
        try:
            messages = [
                SystemMessage(content="You are a medical data analyst. Analyze queries and return only the MINIMUM required fields to answer the question efficiently."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.llm.invoke(messages)
            analysis = json.loads(response.content)
            return analysis
            
        except Exception as e:
            print(f"Error analyzing query: {e}")
            # Fallback to simple keyword analysis
            return self._simple_query_analysis(query)

    def _simple_query_analysis(self, query: str) -> Dict[str, Any]:
        """Simple keyword-based query analysis when AI is not available."""
        query_lower = query.lower()
        
        # Define keyword mappings with specific fields
        keyword_mappings = {
            "demographics": {
                "keywords": ["name", "age", "gender", "address", "contact", "birth", "dob"],
                "fields": ["name", "gender", "dob", "birthDate", "address", "contact"]
            },
            "medications": {
                "keywords": ["medication", "drug", "prescription", "medicine", "taking"],
                "fields": ["genericName", "brandName", "dosage", "startDate", "endDate"]
            },
            "diagnoses": {
                "keywords": ["diagnosis", "condition", "disease", "illness", "icd"],
                "fields": ["code.code", "code.display", "effectiveDateTime"]
            },
            "procedures": {
                "keywords": ["procedure", "surgery", "operation", "treatment"],
                "fields": ["code.display", "performedDateTime", "outcome"]
            },
            "labs": {
                "keywords": ["lab", "test", "result", "blood", "urine", "glucose"],
                "fields": ["code.display", "valueQuantity.value", "valueQuantity.unit", "effectiveDateTime"]
            },
            "allergies": {
                "keywords": ["allergy", "allergic", "reaction", "allergen"],
                "fields": ["allergenname", "reaction", "severity"]
            },
            "encounters": {
                "keywords": ["visit", "encounter", "admission", "emergency", "hospital"],
                "fields": ["status", "class.display", "period.start", "period.end", "reasonCode"]
            }
        }
        
        detected_types = []
        all_fields = []
        
        for data_type, config in keyword_mappings.items():
            if any(keyword in query_lower for keyword in config["keywords"]):
                detected_types.append(data_type)
                all_fields.extend(config["fields"])
        
        if not detected_types:
            detected_types = ["demographics"]  # Default to basic info instead of comprehensive
            all_fields = ["name", "gender", "dob"]
        
        return {
            "query_type": detected_types[0] if len(detected_types) == 1 else detected_types[0],
            "specific_fields": all_fields,
            "data_types": detected_types,
            "scope": "single_patient",
            "search_criteria": {},
            "confidence": 0.6,
            "max_records_per_type": 5
        }

    def _extract_relevant_data(self, query_analysis: Dict[str, Any], patient_id: str = None) -> Dict[str, Any]:
        """Extract relevant patient data based on the query analysis."""
        
        if not patient_id and query_analysis.get("scope") == "single_patient":
            # If no patient ID provided but single patient query, return available patient IDs
            return {"available_patients": self.get_patient_ids()}
        
        if patient_id:
            # Extract data for specific patient
            return self._get_patient_data_by_type(patient_id, query_analysis)
        else:
            # Handle multi-patient queries or searches
            return self._handle_multi_patient_query(query_analysis)

    def _extract_specific_fields(self, data: List[Dict], fields: List[str], max_records: int = 5) -> List[Dict]:
        """Extract only specific fields from data records to minimize context usage."""
        if not data or not fields:
            return []
        
        # Limit number of records to prevent context overflow
        limited_data = data[:max_records]
        extracted = []
        
        for record in limited_data:
            extracted_record = {}
            for field in fields:
                # Handle nested field access (e.g., "code.display")
                if '.' in field:
                    parts = field.split('.')
                    value = record
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    if value is not None:
                        extracted_record[field] = value
                else:
                    # Simple field access
                    if field in record:
                        extracted_record[field] = record[field]
            
            if extracted_record:  # Only add if we found some data
                extracted.append(extracted_record)
        
        return extracted

    def _get_patient_data_by_type(self, patient_id: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get specific types of data for a patient based on query analysis, extracting only needed fields."""
        
        data = {}
        data_types = query_analysis.get("data_types", [query_analysis.get("query_type", "demographics")])
        specific_fields = query_analysis.get("specific_fields", [])
        max_records = query_analysis.get("max_records_per_type", 5)
        
        # Map data types to their corresponding getter methods and field mappings
        data_type_mapping = {
            "demographics": {
                "getter": self.get_patient_demographics,
                "default_fields": ["name", "gender", "dob", "birthDate"]
            },
            "medications": {
                "getter": self.get_patient_medications,
                "default_fields": ["genericName", "brandName", "dosage", "startDate", "endDate"]
            },
            "diagnoses": {
                "getter": self.get_patient_diagnoses,
                "default_fields": ["code.code", "code.display", "effectiveDateTime"]
            },
            "procedures": {
                "getter": self.get_patient_procedures,
                "default_fields": ["code.display", "performedDateTime", "outcome"]
            },
            "labs": {
                "getter": self.get_patient_labs,
                "default_fields": ["code.display", "valueQuantity.value", "valueQuantity.unit", "effectiveDateTime"]
            },
            "allergies": {
                "getter": self.get_patient_allergies,
                "default_fields": ["allergenname", "reaction", "severity"]
            },
            "encounters": {
                "getter": self.get_patient_encounters,
                "default_fields": ["status", "class.display", "period.start", "period.end", "reasonCode"]
            },
            "visits": {
                "getter": self.get_patient_visits,
                "default_fields": ["visitType", "period.start", "period.end", "status"]
            }
        }
        
        for data_type in data_types:
            if data_type in data_type_mapping:
                config = data_type_mapping[data_type]
                raw_data = config["getter"](patient_id)
                
                if data_type == "demographics":
                    # Demographics is a single dict, not a list
                    if raw_data:
                        # Extract specific fields for demographics
                        relevant_fields = [f for f in specific_fields if any(df in f for df in config["default_fields"])]
                        if not relevant_fields:
                            relevant_fields = config["default_fields"]
                        
                        extracted = {}
                        for field in relevant_fields:
                            if field in raw_data:
                                extracted[field] = raw_data[field]
                        data[data_type] = extracted
                else:
                    # Other data types are lists
                    if raw_data:
                        # Filter fields relevant to this data type
                        relevant_fields = [f for f in specific_fields if any(df in f for df in config["default_fields"])]
                        if not relevant_fields:
                            relevant_fields = config["default_fields"]
                        
                        data[data_type] = self._extract_specific_fields(raw_data, relevant_fields, max_records)
        
        return data

    def _handle_multi_patient_query(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries that involve multiple patients or searches."""
        
        search_criteria = query_analysis.get("search_criteria", {})
        
        if search_criteria:
            # Perform searches based on criteria
            results = {}
            
            if "diagnosis" in search_criteria:
                results["patients_with_diagnosis"] = self.search_by_diagnosis(search_criteria["diagnosis"])
            
            if "medication" in search_criteria:
                results["patients_with_medication"] = self.search_by_medication(search_criteria["medication"])
            
            return results
        else:
            # Return summary data for all patients
            all_patients = self.get_patient_ids()
            return {
                "all_patients": all_patients,
                "patient_summaries": [self.get_patient_summary(pid) for pid in all_patients[:5]]  # Limit to first 5
            }

    def _summarize_data_for_llm(self, relevant_data: Dict[str, Any]) -> str:
        """Summarize the relevant data in a concise format for LLM processing."""
        if not relevant_data:
            return "No relevant patient data found."
        
        summary_parts = []
        
        for data_type, data in relevant_data.items():
            if not data:
                continue
                
            if data_type == "demographics":
                demo_info = []
                for key, value in data.items():
                    demo_info.append(f"{key}: {value}")
                summary_parts.append(f"Demographics: {', '.join(demo_info)}")
                
            elif isinstance(data, list):
                if data:
                    summary_parts.append(f"{data_type.title()}: {len(data)} records")
                    # Show key details from first few records
                    for i, record in enumerate(data[:3]):  # Limit to first 3 records
                        key_details = []
                        for key, value in record.items():
                            if value:  # Only include non-empty values
                                key_details.append(f"{key}={value}")
                        if key_details:
                            summary_parts.append(f"  Record {i+1}: {', '.join(key_details)}")
        
        return "\n".join(summary_parts) if summary_parts else "No relevant patient data found."

    def _generate_response(self, original_query: str, query_analysis: Dict[str, Any], relevant_data: Dict[str, Any]) -> str:
        """Generate a natural language response using the AI model."""
        
        if not self.llm:
            return self._generate_simple_response(original_query, relevant_data)
        
        # Create a concise summary instead of full JSON dump
        data_summary = self._summarize_data_for_llm(relevant_data)
        
        response_prompt = f"""
        Based on the patient data provided, answer the following query in a clear, professional medical context.
        
        Original Query: "{original_query}"
        
        Query Type: {query_analysis.get('query_type', 'general')}
        
        Relevant Patient Data Summary:
        {data_summary}
        
        Instructions:
        1. Provide a direct, clear answer to the user's question
        2. Include relevant medical details and specific data points from the summary
        3. Maintain patient privacy and professional medical language
        4. If data is missing or incomplete, mention this clearly
        5. Format the response in a readable, structured manner
        6. Include relevant dates, values, and medical codes when appropriate
        7. Be concise but comprehensive
        
        Response:
        """
        
        try:
            messages = [
                SystemMessage(content="You are a medical information assistant. Provide clear, accurate responses based on patient data while maintaining professional medical standards."),
                HumanMessage(content=response_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return self._generate_simple_response(original_query, relevant_data)

    def _generate_simple_response(self, query: str, relevant_data: Dict[str, Any]) -> str:
        """Generate a simple response when AI is not available."""
        
        if "available_patients" in relevant_data:
            return f"Available patients: {', '.join(relevant_data['available_patients'])}"
        
        if not relevant_data:
            return "No relevant patient data found for your query."
        
        response_parts = []
        
        for data_type, data in relevant_data.items():
            if data:
                if data_type == "demographics":
                    response_parts.append(f"Demographics: {data}")
                elif data_type == "medications":
                    med_names = [med.get('genericName', 'Unknown') for med in data]
                    response_parts.append(f"Medications: {', '.join(med_names)}")
                elif data_type == "diagnoses":
                    diag_names = []
                    for diag in data:
                        for code in diag.get('code', []):
                            if code.get('display'):
                                diag_names.append(code.get('display'))
                    response_parts.append(f"Diagnoses: {', '.join(diag_names)}")
                else:
                    response_parts.append(f"{data_type.title()}: {len(data)} records found")
        
        return "; ".join(response_parts) if response_parts else "Patient data retrieved successfully."

    def _process_simple_query(self, query: str, patient_id: str = None) -> Dict[str, Any]:
        """Process query without AI - uses simple keyword analysis."""
        query_analysis = self._simple_query_analysis(query)
        relevant_data = self._extract_relevant_data(query_analysis, patient_id)
        response = self._generate_simple_response(query, relevant_data)
        
        return {
            "query": query,
            "patient_id": patient_id,
            "answer": response,
            "relevant_data": relevant_data,
            "query_type": query_analysis.get("query_type"),
            "confidence": query_analysis.get("confidence", 0.6)
        }

    def ask_about_patient(self, question: str, patient_id: str = None) -> str:
        """
        Convenient method to ask questions about a patient and get natural language answers.
        
        Args:
            question (str): Natural language question about the patient
            patient_id (str, optional): Specific patient ID
            
        Returns:
            str: Natural language answer to the question
        """
        result = self.process_query(question, patient_id)
        return result["answer"]

    def get_patient_summary_with_ai(self, patient_id: str) -> str:
        """Get an AI-generated comprehensive summary of a patient."""
        return self.ask_about_patient(
            f"Provide a comprehensive medical summary for patient {patient_id}, including their demographics, medical history, current medications, recent encounters, and any significant health concerns.",
            patient_id
        )

    def get_patient_ids(self) -> List[str]:
        """Get all patient IDs in the dataset."""
        patient_ids = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'patient' in section:
                    patient_ids.append(section['patient']['patient_id'])
        return patient_ids
    
    def get_patient_demographics(self, patient_id: str) -> Optional[Dict]:
        """Get demographic information for a specific patient."""
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'patient' in section and section['patient']['patient_id'] == patient_id:
                    return section['patient']
        return None
    
    def get_patient_encounters(self, patient_id: str) -> List[Dict]:
        """Get all encounters for a specific patient."""
        encounters = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'encounters' in section:
                    for encounter in section['encounters']:
                        if encounter.get('patientId') == patient_id:
                            encounters.append(encounter)
        return encounters
    
    def get_patient_visits(self, patient_id: str) -> List[Dict]:
        """Get all visits for a specific patient."""
        visits = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'visits' in section:
                    for visit in section['visits']:
                        if visit.get('patient_id') == patient_id:
                            visits.append(visit)
        return visits
    
    def get_patient_procedures(self, patient_id: str) -> List[Dict]:
        """Get all procedures for a specific patient."""
        procedures = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'procedures' in section:
                    for procedure in section['procedures']:
                        if procedure.get('patient_id') == patient_id:
                            procedures.append(procedure)
        return procedures
    
    def get_patient_diagnoses(self, patient_id: str) -> List[Dict]:
        """Get all diagnoses for a specific patient."""
        diagnoses = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'diagnosis' in section:
                    for diagnosis in section['diagnosis']:
                        if diagnosis.get('patient_id') == patient_id:
                            diagnoses.append(diagnosis)
        return diagnoses
    
    def get_patient_labs(self, patient_id: str) -> List[Dict]:
        """Get all lab results for a specific patient."""
        labs = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'labs' in section:
                    for lab in section['labs']:
                        if lab.get('patient_id') == patient_id:
                            labs.append(lab)
        return labs
    
    def get_patient_medications(self, patient_id: str) -> List[Dict]:
        """Get all medications for a specific patient."""
        medications = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'medications' in section:
                    for medication in section['medications']:
                        if medication.get('patient_id') == patient_id:
                            medications.append(medication)
        return medications
    
    def get_patient_allergies(self, patient_id: str) -> List[Dict]:
        """Get all allergies for a specific patient."""
        allergies = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'allergies' in section:
                    for allergy in section['allergies']:
                        if allergy.get('patient_id') == patient_id:
                            allergies.append(allergy)
        return allergies
    
    def get_patient_notes(self, patient_id: str) -> List[Dict]:
        """Get all clinical notes for a specific patient."""
        notes = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'notes' in section:
                    for note in section['notes']:
                        # Notes don't have patient_id directly, need to match by context
                        # Assuming notes are grouped with their respective patients
                        notes.append(note)
        return notes
    
    def get_complete_patient_record(self, patient_id: str) -> Dict:
        """Get complete medical record for a specific patient."""
        return {
            'demographics': self.get_patient_demographics(patient_id),
            'encounters': self.get_patient_encounters(patient_id),
            'visits': self.get_patient_visits(patient_id),
            'procedures': self.get_patient_procedures(patient_id),
            'diagnoses': self.get_patient_diagnoses(patient_id),
            'labs': self.get_patient_labs(patient_id),
            'medications': self.get_patient_medications(patient_id),
            'allergies': self.get_patient_allergies(patient_id),
            'notes': self.get_patient_notes(patient_id)
        }
    
    def search_by_diagnosis(self, diagnosis_code: str) -> List[str]:
        """Search for patients with a specific diagnosis code."""
        matching_patients = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'diagnosis' in section:
                    for diagnosis in section['diagnosis']:
                        for code in diagnosis.get('code', []):
                            if code.get('code') == diagnosis_code:
                                patient_id = diagnosis.get('patient_id')
                                if patient_id and patient_id not in matching_patients:
                                    matching_patients.append(patient_id)
        return matching_patients
    
    def search_by_medication(self, medication_name: str) -> List[str]:
        """Search for patients taking a specific medication."""
        matching_patients = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'medications' in section:
                    for medication in section['medications']:
                        if medication.get('genericName', '').lower() == medication_name.lower():
                            patient_id = medication.get('patient_id')
                            if patient_id and patient_id not in matching_patients:
                                matching_patients.append(patient_id)
        return matching_patients
    
    def get_patient_summary(self, patient_id: str) -> Dict:
        """Generate a summary of a patient's medical history."""
        demographics = self.get_patient_demographics(patient_id)
        encounters = self.get_patient_encounters(patient_id)
        diagnoses = self.get_patient_diagnoses(patient_id)
        medications = self.get_patient_medications(patient_id)
        allergies = self.get_patient_allergies(patient_id)
        
        if not demographics:
            return {"error": f"Patient {patient_id} not found"}
        
        # Extract key information
        name = demographics.get('name', 'Unknown')
        if isinstance(name, dict):
            name = f"{name.get('given', '')} {name.get('family', '')}"
        
        gender = demographics.get('gender', 'Unknown')
        dob = demographics.get('dob') or demographics.get('birthDate', 'Unknown')
        
        # Count various records
        total_encounters = len(encounters)
        total_diagnoses = len(diagnoses)
        active_medications = len([med for med in medications 
                                if med.get('endDate', '').lower() in ['present', '2024-12-31', '']])
        total_allergies = len(allergies)
        
        # Extract primary diagnoses
        primary_diagnoses = []
        for diagnosis in diagnoses:
            for code in diagnosis.get('code', []):
                if code.get('display'):
                    primary_diagnoses.append(code.get('display'))
        
        return {
            'patient_id': patient_id,
            'name': name,
            'gender': gender,
            'date_of_birth': dob,
            'total_encounters': total_encounters,
            'total_diagnoses': total_diagnoses,
            'active_medications': active_medications,
            'total_allergies': total_allergies,
            'primary_diagnoses': list(set(primary_diagnoses)),
            'medication_list': [med.get('genericName') for med in medications],
            'allergy_list': [allergy.get('allergenname') for allergy in allergies]
        }

    def export_patient_data_to_csv(self, patient_id: str, output_dir: str = './'):
        """
        Export patient data to CSV files for analysis.
        
        Args:
            patient_id (str): The patient ID
            output_dir (str): Directory to save CSV files
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all patient data
        encounters = self.get_patient_encounters(patient_id)
        visits = self.get_patient_visits(patient_id)
        procedures = self.get_patient_procedures(patient_id)
        diagnoses = self.get_patient_diagnoses(patient_id)
        labs = self.get_patient_labs(patient_id)
        medications = self.get_patient_medications(patient_id)
        allergies = self.get_patient_allergies(patient_id)
        
        # Convert to DataFrames and save
        data_types = {
            'encounters': encounters,
            'visits': visits,
            'procedures': procedures,
            'diagnoses': diagnoses,
            'labs': labs,
            'medications': medications,
            'allergies': allergies
        }
        
        for data_type, data in data_types.items():
            if data:
                try:
                    df = pd.json_normalize(data)
                    filename = f"{patient_id}_{data_type}.csv"
                    filepath = os.path.join(output_dir, filename)
                    df.to_csv(filepath, index=False)
                    print(f"Exported {len(data)} {data_type} records to {filepath}")
                except Exception as e:
                    print(f"Error exporting {data_type}: {str(e)}")

    def print_patient_summary(self, patient_id: str):
        """Print a formatted summary of a patient's medical history."""
        summary = self.get_patient_summary(patient_id)
        
        if 'error' in summary:
            print(summary['error'])
            return
        
        print("="*60)
        print(f"PATIENT SUMMARY: {summary['name']} ({summary['patient_id']})")
        print("="*60)
        print(f"Gender: {summary['gender']}")
        print(f"Date of Birth: {summary['date_of_birth']}")
        print(f"Total Encounters: {summary['total_encounters']}")
        print(f"Total Diagnoses: {summary['total_diagnoses']}")
        print(f"Active Medications: {summary['active_medications']}")
        print(f"Total Allergies: {summary['total_allergies']}")
        
        print("\nPrimary Diagnoses:")
        for diagnosis in summary['primary_diagnoses']:
            print(f"  - {diagnosis}")
        
        print("\nCurrent Medications:")
        for medication in summary['medication_list']:
            print(f"  - {medication}")
        
        print("\nAllergies:")
        for allergy in summary['allergy_list']:
            print(f"  - {allergy}")
        print("="*60)
