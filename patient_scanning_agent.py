import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

class PatientScanningAgent:
    """
    A comprehensive patient scanning agent that can read and parse complex patient records
    from a JSON file containing multiple patients with nested medical data.
    """
    
    def __init__(self, json_file_path: str):
        """
        Initialize the agent with a JSON file containing patient records.
        
        Args:
            json_file_path (str): Path to the patients.json file
        """
        self.json_file_path = json_file_path
        self.patients_data = []
        self.load_patient_data()
    
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
    
    def get_patient_ids(self) -> List[str]:
        """Get all patient IDs in the dataset."""
        patient_ids = []
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'patient' in section:
                    patient_ids.append(section['patient']['patient_id'])
        return patient_ids
    
    def get_patient_demographics(self, patient_id: str) -> Optional[Dict]:
        """
        Get demographic information for a specific patient.
        
        Args:
            patient_id (str): The patient ID to search for
            
        Returns:
            Dict: Patient demographic information or None if not found
        """
        for patient_record in self.patients_data:
            for section in patient_record:
                if 'patient' in section and section['patient']['patient_id'] == patient_id:
                    return section['patient']
        return None
    
    def get_patient_encounters(self, patient_id: str) -> List[Dict]:
        """
        Get all encounters for a specific patient.
        
        Args:
            patient_id (str): The patient ID to search for
            
        Returns:
            List[Dict]: List of encounters for the patient
        """
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
        """
        Get complete medical record for a specific patient.
        
        Args:
            patient_id (str): The patient ID to search for
            
        Returns:
            Dict: Complete patient record with all medical data
        """
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
        """
        Search for patients with a specific diagnosis code.
        
        Args:
            diagnosis_code (str): ICD-10 or other diagnosis code
            
        Returns:
            List[str]: List of patient IDs with the diagnosis
        """
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
        """
        Search for patients taking a specific medication.
        
        Args:
            medication_name (str): Generic name of the medication
            
        Returns:
            List[str]: List of patient IDs taking the medication
        """
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
        """
        Generate a summary of a patient's medical history.
        
        Args:
            patient_id (str): The patient ID
            
        Returns:
            Dict: Summary of patient's medical history
        """
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


def main():
    """Example usage of the PatientScanningAgent."""
    # Initialize the agent
    agent = PatientScanningAgent('./patients.json')
    
    # Get all patient IDs
    patient_ids = agent.get_patient_ids()
    print(f"Found {len(patient_ids)} patients: {patient_ids}")
    
    # Print summaries for all patients
    for patient_id in patient_ids:
        agent.print_patient_summary(patient_id)
        print("\n")
    
    # Example searches
    print("Patients with lung cancer:")
    lung_cancer_patients = agent.search_by_diagnosis("C34.1")
    print(lung_cancer_patients)
    
    print("\nPatients taking Metformin:")
    metformin_patients = agent.search_by_medication("Metformin")
    print(metformin_patients)
    
    # Export patient data to CSV (uncomment to use)
    # if patient_ids:
    #     agent.export_patient_data_to_csv(patient_ids[0], './patient_exports/')


if __name__ == "__main__":
    main() 