#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative AI Data Cards Generator with Confidence Scoring - Optimized Version
"""

import json
import os
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generative_ai_data_cards.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GenerativeAIDataCardGenerator:
    def __init__(self, api_key: str, max_workers: int = 100):
        self.api_key = api_key
        self.max_workers = max_workers
        self.session = None
        
        # Optimized template for generative AI data cards
        self.generative_ai_data_template = """
You are an expert in Generative AI dataset documentation, specializing in creating comprehensive data cards for GENERATIVE AI TRAINING DATASETS.

IMPORTANT INSTRUCTIONS:
1. If you cannot find specific information about any section, simply output "Not provided" and STOP. Do not add any explanatory text.
2. If a field doesn't apply to this type of dataset, simply output "Not applicable" and STOP. Do not add any explanatory text.
3. Be concise and factual. Only extract information that is explicitly stated.
4. For each section, provide a confidence level: "low", "medium", "high", or "certain".
5. Return VALID JSON format with each section containing both "content" and "confidence" fields.

Based on the following information:
- Dataset ID: {dataset_id}
- Tags: {tags}
- Dataset Card: {dataset_card}
- Description: {description}

Create a data card and return it as a JSON object with the following structure:

{{
  "dataset_details": {{
    "dataset_name": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "version": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "creators_curators": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "funding": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "dataset_type": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "text_language": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "license": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "related_resources": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "dataset_structure": {{
    "instances": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "fields": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "missing_info": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "relationships": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "splits": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "size_statistics": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "data_collection": {{
    "collection_process": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "data_sources": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "collection_timeframe": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "ethical_review": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "consent_process": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "data_validation": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "data_processing": {{
    "preprocessing_steps": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "cleaning_procedures": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "labeling_process": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "quality_control": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "filtering_criteria": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "deduplication": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "intended_uses": {{
    "primary_tasks": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "suitable_applications": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "unsuitable_applications": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "research_applications": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "commercial_applications": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "prohibited_uses": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "bias_and_fairness": {{
    "demographic_representation": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "geographic_coverage": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "temporal_coverage": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "known_biases": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "bias_mitigation": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "fairness_considerations": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "privacy_and_security": {{
    "personally_identifiable_info": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "sensitive_information": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "privacy_protection_measures": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "data_security": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "anonymization_pseudonymization": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "retention_deletion": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "content_analysis": {{
    "content_types": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "harmful_content": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "content_moderation": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "toxicity_analysis": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "misinformation_risks": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "cultural_sensitivity": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "legal_and_ethical": {{
    "copyright_considerations": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "terms_of_use": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "ethical_guidelines": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "compliance_requirements": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "subject_rights": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "institutional_review": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "maintenance_and_updates": {{
    "maintenance_plan": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "update_frequency": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "versioning": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "error_reporting": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "community_contribution": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "deprecation_plan": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "distribution_and_access": {{
    "access_mechanism": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "distribution_format": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "download_instructions": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "api_access": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "access_restrictions": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "citation_requirements": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "limitations_and_recommendations": {{
    "known_limitations": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "recommended_uses": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "usage_guidelines": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "performance_considerations": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "environmental_impact": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "future_work": {{"content": "content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }}
}}

CRITICAL REMINDERS:
1. Only extract information explicitly stated in the provided data
2. Use "Not provided" for missing information - DO NOT elaborate or explain
3. Use "Not applicable" for irrelevant fields - DO NOT elaborate or explain  
4. Return ONLY the JSON object, no additional text
5. Ensure all JSON is properly formatted and valid
"""

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=500, limit_per_host=100)
        timeout = aiohttp.ClientTimeout(total=300)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def determine_dataset_type(self, tags: Union[str, List[str], None], dataset_id: str) -> str:
        """Determine dataset type based on tags and dataset ID"""
        if tags is None:
            tags_str = ""
        elif isinstance(tags, list):
            tags_str = " ".join(str(tag) for tag in tags)
        else:
            tags_str = str(tags)
        
        tags_lower = tags_str.lower()
        dataset_id_lower = dataset_id.lower()
        
        if any(term in tags_lower for term in ['multimodal', 'vision-language', 'text-image']):
            return 'multimodal-dataset'
        elif any(term in tags_lower for term in ['image', 'visual', 'computer-vision', 'cv']):
            return 'image-dataset'
        elif any(term in tags_lower for term in ['audio', 'speech', 'music', 'sound']):
            return 'audio-dataset'
        elif any(term in tags_lower for term in ['video', 'temporal', 'sequence']):
            return 'video-dataset'
        elif any(term in tags_lower for term in ['code', 'programming', 'software']):
            return 'code-dataset'
        elif any(term in tags_lower for term in ['conversation', 'dialogue', 'chat']):
            return 'conversational-dataset'
        else:
            return 'text-dataset'

    def filter_relevant_data(self, data: Dict) -> Dict:
        """Filter out irrelevant data like RepoSibling and keep only essential fields"""
        filtered_data = {
            'datasetId': data.get('datasetId', ''),
            'datasetCard': data.get('datasetCard', ''),
            'tags': data.get('tags', []),
            'description': data.get('description', ''),
            'author': data.get('author', ''),
            'likes': data.get('likes', 0),
            'downloads': data.get('downloads', 0),
            'lastModified': data.get('lastModified', ''),
            'createdAt': data.get('createdAt', ''),
            'generative_ai_info': data.get('generative_ai_info', {})
        }
        
        # Extract license from metadata if available
        if 'metadata' in data and 'card_data' in data['metadata']:
            card_data = data['metadata']['card_data']
            if 'license' in card_data:
                filtered_data['license'] = card_data.get('license', '')
        
        return filtered_data

    def parse_generated_card(self, generated_text: str) -> Dict:
        """Parse generated data card JSON text"""
        try:
            # Try to parse JSON directly
            card_json = json.loads(generated_text)
            return card_json
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON part
            try:
                # Look for JSON start and end markers
                start_idx = generated_text.find('{')
                end_idx = generated_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_part = generated_text[start_idx:end_idx]
                    card_json = json.loads(json_part)
                    return card_json
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                # If still fails, return raw text as fallback
                return {
                    "raw_content": {
                        "content": generated_text,
                        "confidence": "low"
                    }
                }

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        base=2,
        max_value=60
    )
    async def generate_data_card(self, dataset_data: Dict) -> Optional[Dict]:
        """Generate specialized data card for generative AI datasets"""
        
        # Filter relevant data only
        filtered_data = self.filter_relevant_data(dataset_data)
        
        # Get the dataset_id for logging
        dataset_id = filtered_data.get('datasetId', 'Unknown')
        logger.info(f"Processing dataset: {dataset_id}")
        
        # Determine dataset type
        dataset_type = self.determine_dataset_type(
            filtered_data.get('tags'), 
            dataset_id
        )
        
        # Convert tags to string for display
        tags_display = filtered_data.get('tags', [])
        if isinstance(tags_display, list):
            tags_display = ", ".join(str(tag) for tag in tags_display) if tags_display else "No tags available"
        elif tags_display is None:
            tags_display = "No tags available"
        else:
            tags_display = str(tags_display)
        
        prompt = self.generative_ai_data_template.format(
            dataset_id=dataset_id,
            tags=tags_display,
            dataset_card=str(filtered_data.get('datasetCard', '')),
            description=str(filtered_data.get('description', ''))
        )
        
        payload = {
            "model": "gpt-5-mini-2025-08-07",
            "messages": [
                {
                    "role": "system", 
                    "content": f"You are an expert in generative AI dataset documentation. Be concise and factual. When information is missing, simply output 'Not provided'. When irrelevant, simply output 'Not applicable'. Do not elaborate."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_completion_tokens": 8000
        }
        
        try:
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    generated_text = result['choices'][0]['message']['content']
                    
                    # Parse generated JSON format data card
                    parsed_card = self.parse_generated_card(generated_text)
                    
                    # Return only dataset ID and generated card
                    return {
                        'datasetId': dataset_id,
                        'generated_data_card': parsed_card,
                        'generation_status': 'success',
                        'generation_timestamp': datetime.now().isoformat()
                    }
                    
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status} for dataset {dataset_id}: {error_text}")
                    
                    return {
                        'datasetId': dataset_id,
                        'generation_status': 'failed',
                        'generation_error': f"API error {response.status}: {error_text}",
                        'generation_timestamp': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Exception generating card for {dataset_id}: {e}")
            
            return {
                'datasetId': dataset_id,
                'generation_status': 'failed',
                'generation_error': str(e),
                'generation_timestamp': datetime.now().isoformat()
            }

    def read_and_filter_data(self, file_path: str) -> List[Dict]:
        """Read and filter generative AI datasets"""
        logger.info(f"Reading generative AI datasets from: {file_path}")
        generative_datasets = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # Only process datasets marked as generative AI
                        if data.get('generative_ai_info', {}).get('is_generative_ai', False):
                            generative_datasets.append(data)
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
            
        logger.info(f"Found {len(generative_datasets)} generative AI datasets")
        
        # Sort by popularity (downloads + likes)
        generative_datasets.sort(
            key=lambda x: x.get('downloads', 0) + x.get('likes', 0), 
            reverse=True
        )
        
        return generative_datasets

    async def process_batch_async(self, datasets_batch: List[Dict]) -> List[Dict]:
        """Async batch processing of datasets"""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single(dataset_data):
            async with semaphore:
                return await self.generate_data_card(dataset_data)
        
        tasks = [process_single(dataset) for dataset in datasets_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                dataset_id = datasets_batch[i].get('datasetId', 'Unknown')
                logger.error(f"Task failed for {dataset_id}: {result}")
                processed_results.append({
                    'datasetId': dataset_id,
                    'generation_status': 'failed',
                    'generation_error': str(result),
                    'generation_timestamp': datetime.now().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results

    def save_checkpoint(self, results: List[Dict], output_path: str, checkpoint_num: int):
        """Save checkpoint"""
        checkpoint_path = f"{output_path}.checkpoint_{checkpoint_num}.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, output_path: str) -> List[Dict]:
        """Load latest checkpoint"""
        try:
            checkpoint_dir = os.path.dirname(output_path) or '.'
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                               if f.startswith(os.path.basename(output_path) + '.checkpoint_')]
            if not checkpoint_files:
                return []
            
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return []

    async def process_all_datasets(self, input_file: str, output_file: str, batch_size: int = 50):
        """Process all generative AI datasets"""
        # Read and filter data
        all_datasets = self.read_and_filter_data(input_file)
        
        # Check checkpoint
        processed_results = self.load_checkpoint(output_file)
        processed_ids = {r.get('datasetId') for r in processed_results 
                        if r.get('datasetId') is not None and r.get('datasetId') != ''}
        
        # Filter already processed datasets
        remaining_datasets = [d for d in all_datasets 
                             if d.get('datasetId') not in processed_ids]
        
        logger.info(f"Total generative AI datasets: {len(all_datasets)}, Remaining: {len(remaining_datasets)}")
        
        # Process in batches
        checkpoint_interval = 5  # Save checkpoint every 5 batches
        
        for i in range(0, len(remaining_datasets), batch_size):
            batch = remaining_datasets[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{(len(remaining_datasets) + batch_size - 1) // batch_size}")
            
            # Process current batch
            batch_results = await self.process_batch_async(batch)
            processed_results.extend(batch_results)
            
            # Periodically save checkpoint
            if batch_num % checkpoint_interval == 0:
                self.save_checkpoint(processed_results, output_file, batch_num)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Save final results - only dataset ID and generated cards
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processing complete. Results saved to: {output_file}")
        
        # Generate summary report
        self.generate_summary_report(processed_results, output_file + '.summary.json')
        
        return processed_results

    def generate_summary_report(self, results: List[Dict], report_path: str):
        """Generate summary report"""
        total = len(results)
        successful = len([r for r in results if r.get('generation_status') == 'success'])
        failed = total - successful
        
        report = {
            'summary': {
                'total_datasets': total,
                'successful': successful,
                'failed': failed,
                'success_rate': f"{(successful/total*100):.2f}%" if total > 0 else "0%",
                'timestamp': datetime.now().isoformat()
            },
            'failed_datasets': [
                {
                    'datasetId': r.get('datasetId', 'unknown'), 
                    'error': r.get('generation_error', 'unknown')
                } 
                for r in results if r.get('generation_status') == 'failed'
            ]
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Summary report generated: {report_path}")
        logger.info(f"Success rate: {report['summary']['success_rate']} ({successful}/{total})")

async def main():
    """Main function"""
    API_KEY = os.getenv('OPENAI_API_KEY')
    if not API_KEY:
        logger.error("Please set OPENAI_API_KEY environment variable")
        return
    
    INPUT_FILE = "/mnt/shared_disk/datacards/dataset/processed/cleaned_dataset_cards_with_generative_ai.jsonl"  # 您的输入文件路径
    OUTPUT_FILE = "/mnt/shared_disk/datacards/dataset/processed/generated_data_cards.json"  # 输出文件路径
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Use optimized generator
    async with GenerativeAIDataCardGenerator(API_KEY, max_workers=100) as generator:
        try:
            results = await generator.process_all_datasets(INPUT_FILE, OUTPUT_FILE, batch_size=30)
            logger.info("Generative AI Data Cards generation completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted. Progress saved in checkpoint files.")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())
