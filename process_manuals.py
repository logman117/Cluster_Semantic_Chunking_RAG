import os
import asyncio
import pandas as pd
import shutil
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from cluster_semantic_chunker import Chunk, process_documents
from embedding_generation import EmbeddingGenerator, SupabaseVectorStore, process_and_store

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def load_manual_machine_mapping(excel_path: str) -> Dict[str, Dict]:
    """
    Load mapping between service manual filenames and machine IDs from Excel.
    
    Args:
        excel_path: Path to the Excel file containing the mapping
        
    Returns:
        Dictionary mapping manual filenames to machine info
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path, sheet_name='Sheet1')
        
        # Create mapping: filename -> machine_info
        manual_to_machine = {}
        
        # Assuming columns are named appropriately
        for _, row in df.iterrows():
            manual_filename = row['Service Manual Name']
            
            # Convert any value to string first to handle both text and numeric data
            model_number = str(row['Correlated Model Number'])
            machine_id = model_number.lower().replace(' ', '_')
            
            machine_name = row['Correlated Model Description']
            
            # Add to mapping
            if manual_filename and machine_id:
                manual_to_machine[manual_filename] = {
                    'id': machine_id,
                    'name': machine_name,
                    'model': model_number
                }
        
        return manual_to_machine
    except Exception as e:
        logger.error(f"Error loading manual-machine mapping: {e}")
        return {}

async def process_pdf_for_machine(pdf_path: str, machine_info: Dict) -> None:
    """Process a single PDF file for a specific machine."""
    
    # Create a temporary directory
    temp_dir = os.path.join(os.getcwd(), "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Copy the PDF to the temporary directory
        filename = os.path.basename(pdf_path)
        temp_pdf_path = os.path.join(temp_dir, filename)
        shutil.copy(pdf_path, temp_pdf_path)
        
        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator()
        
        # Process the document
        chunks = process_documents(temp_dir, embedding_gen.generate_embedding_sync)
        
        if not chunks:
            logger.warning(f"No chunks generated for {filename}")
            return
        
        logger.info(f"Generated {len(chunks)} chunks from {filename}")
        
        # Get the machine ID
        machine_id = machine_info['id']
        
        # Verify machine exists or create it
        vector_store = SupabaseVectorStore()
        db_machine_info = await vector_store.get_machine_info(machine_id)
        
        if not db_machine_info:
            logger.info(f"Machine {machine_id} not found in database. Creating a new machine entry.")
            vector_store.store_machine(
                machine_id=machine_id,
                name=machine_info['name'],
                model=machine_info['model'],
                description=f"Service manual: {filename}"
            )
        
        # Generate embeddings and store in Supabase
        await process_and_store(chunks, machine_id)
        logger.info(f"Successfully processed {filename} for machine {machine_id}")
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

async def main():
    """Process all PDF files according to the Excel mapping."""
    
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Paths
    pdf_dir = os.path.join(current_dir, "data", "pdfs")
    mapping_file = os.path.join(current_dir, "Service_manual_L5_relations.xlsx")
    
    # Make sure directories exist
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Check if mapping file exists
    if not os.path.exists(mapping_file):
        logger.error(f"Mapping file not found: {mapping_file}")
        return
    
    # Load the mapping
    mapping = await load_manual_machine_mapping(mapping_file)
    logger.info(f"Loaded mapping for {len(mapping)} manuals")
    
    # List PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files in directory")
    
    # Process each PDF file with its corresponding machine ID
    for filename in pdf_files:
        if filename in mapping:
            logger.info(f"Processing {filename} for machine {mapping[filename]['id']}")
            pdf_path = os.path.join(pdf_dir, filename)
            await process_pdf_for_machine(pdf_path, mapping[filename])
        else:
            logger.warning(f"Skipping {filename} - not found in mapping file")

if __name__ == "__main__":
    asyncio.run(main())