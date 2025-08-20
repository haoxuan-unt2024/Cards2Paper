#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative AI Model Cards Generator with Confidence Scoring
"""


import json
import os
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import backoff
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generative_ai_model_cards.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GenerativeAIModelCardGenerator:
    def __init__(self, api_key: str, max_workers: int = 100):
        self.api_key = api_key
        self.max_workers = max_workers
        self.session = None
        
        # Template for generative AI model cards with confidence scoring
        self.generative_ai_template = """
You are an expert in Generative AI model documentation, specializing in creating comprehensive model cards for GENERATIVE AI SYSTEMS.


IMPORTANT INSTRUCTIONS:
1. If you cannot find specific information about any section from the provided data, set content to "Not provided". If a field doesn't apply to this type of model, use "Not applicable". DO NOT make up, invent, or fabricate any information.
2. For each section, provide a confidence level: "low", "medium", "high", or "certain" based on how confident you are about the information.
3. Return your response in VALID JSON format with each section containing both "content" and "confidence" fields.
4. CRITICAL: If the content for any field is "Not provided" or "Not applicable", DO NOT add any additional explanatory text. Just use the exact phrases "Not provided" or "Not applicable" as the content value.


Based on the following information:
- Model ID: {model_id}
- Tags: {tags}
- Original Model Card: {original_card}


Create a comprehensive Model Card specifically designed for this GENERATIVE AI model and return it as a JSON object with the following structure:


{{
  "model_details": {{
    "developer_information": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "model_architecture": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "model_size": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "training_methodology": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "modalities": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "version_information": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "license": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "citation": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "contact": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "intended_use": {{
    "primary_applications": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "target_users": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "supported_languages_domains": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "out_of_scope_uses": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "age_restrictions": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "generative_capabilities": {{
    "generation_quality": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "content_types": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "length_limitations": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "consistency": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "latency": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "customization": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "safety_considerations": {{
    "content_safety": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "bias_analysis": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "fairness_metrics": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "red_team_testing": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "jailbreaking_resistance": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "child_safety": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "training_data": {{
    "training_corpus": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "data_filtering": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "demographic_representation": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "language_coverage": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "consent_privacy": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "evaluation_datasets": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "performance_metrics": {{
    "generation_quality_metrics": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "safety_metrics": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "factual_accuracy": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "bias_metrics": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "cultural_sensitivity": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "robustness": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "ethical_considerations": {{
    "dual_use_risks": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "misinformation": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "intellectual_property": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "economic_impact": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "environmental_impact": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "cultural_appropriation": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "privacy": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "consent": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }},
  "caveats_recommendations": {{
    "known_limitations": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "deployment_recommendations": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "monitoring_requirements": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}},
    "user_guidelines": {{"content": "text content or 'Not provided' or 'Not applicable'", "confidence": "low|medium|high|certain"}}
  }}
}}


CONFIDENCE LEVEL GUIDELINES:
- "certain": Information explicitly and clearly stated in the provided data
- "high": Information that can be reliably inferred from the provided data with high confidence
- "medium": Information based on reasonable assumptions from limited available data
- "low": General recommendations, very uncertain inferences, or when content is "Not provided"/"Not applicable"


CONTENT GUIDELINES:
- "Not provided": Use when specific information should exist but is missing from the source data
- "Not applicable": Use when a field doesn't apply to this type of model (e.g., video generation capabilities for a text-only model)


CRITICAL REMINDERS:
1. Only use information from the provided data
2. DO NOT fabricate, invent, or hallucinate any information
3. Be conservative with confidence levels - when in doubt, use "low"
4. Use "Not provided" when specific information is missing
5. Use "Not applicable" when a field doesn't apply to this type of model
6. Return ONLY the JSON object, no additional text
7. Ensure all JSON is properly formatted and valid
8. When content is "Not provided" or "Not applicable", do NOT add any explanatory text beyond these exact phrases


Generate a professional, factual model card based strictly on the available information.
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


    def normalize_data_structure(self, data: Dict) -> Dict:
        """Normalize data structure, ensure all fields exist with empty defaults"""
        normalized_data = {
            # Basic information - 直接使用源数据的字段
            'model_id': data.get('model_id', ''),  # 直接获取，不做复杂处理
            'id': data.get('id', ''),
            'author': data.get('author', ''),
            'sha': data.get('sha', ''),
            'created_at': data.get('created_at', ''),
            'last_modified': data.get('last_modified', ''),
            'private': data.get('private', False),
            'gated': data.get('gated', False),
            'disabled': data.get('disabled', False),
            'library_name': data.get('library_name', ''),
            'pipeline_tag': data.get('pipeline_tag', ''),
            'tags': data.get('tags', []),
            'downloads': data.get('downloads', 0),
            'downloads_all_time': data.get('downloads_all_time', 0),
            'likes': data.get('likes', 0),
            'license': data.get('license', ''),
            
            # Model card information
            'modelCard': data.get('modelCard', ''),
            'cardData': data.get('cardData', {}),
            
            # File information
            'siblings': data.get('siblings', []),
            'spaces': data.get('spaces', []),
            'safetensors': data.get('safetensors', {}),
            
            # Generative AI information
            'generative_ai_info': data.get('generative_ai_info', {
                'is_generative_ai': False,
                'generative_tasks': [],
                'confidence_score': 0.0,
                'analysis_details': {}
            }),
            
            # Other possible fields
            'config': data.get('config', {}),
            'transformersInfo': data.get('transformersInfo', {}),
            'trending_score': data.get('trending_score', 0.0),
            'inference': data.get('inference', ''),
            'mask_token': data.get('mask_token', ''),
            'widget': data.get('widget', []),
        }
        
        return normalized_data


    def parse_generated_card(self, generated_text: str) -> Dict:
        """Parse generated model card JSON text"""
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
    async def generate_model_card(self, model_data: Dict) -> Optional[Dict]:
        """Generate specialized model card for generative AI"""
        
        # First normalize data structure
        normalized_data = self.normalize_data_structure(model_data)
        
        # Get the model_id for logging
        model_id = normalized_data.get('model_id', 'Unknown')
        logger.info(f"Processing model: {model_id}")
        
        # Convert tags to string for display
        tags_display = normalized_data.get('tags', [])
        if isinstance(tags_display, list):
            tags_display = ", ".join(str(tag) for tag in tags_display) if tags_display else "No tags available"
        elif tags_display is None:
            tags_display = "No tags available"
        else:
            tags_display = str(tags_display)
        
        prompt = self.generative_ai_template.format(
            model_id=model_id,
            tags=tags_display,
            original_card=str(normalized_data.get('modelCard', ''))
        )
        
        payload = {
            "model": "gpt-5-mini-2025-08-07",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert in generative AI documentation, specializing in creating factual model cards for generative AI systems. NEVER fabricate information. Always provide confidence levels. Use 'Not provided' for missing information and 'Not applicable' for irrelevant fields. When using 'Not provided' or 'Not applicable', use ONLY these exact phrases without any additional explanation. Return only valid JSON format with content and confidence fields."
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
                    
                    # Parse generated JSON format model card
                    parsed_card = self.parse_generated_card(generated_text)
                    
                    # Create simplified result with only essential fields
                    simplified_result = {
                        'model_id': model_id,
                        'generated_model_card': parsed_card
                    }
                    
                    logger.info(f"Successfully processed model: {model_id}")
                    return simplified_result
                    
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status} for model {model_id}: {error_text}")
                    
                    # Return simplified result even on failure
                    simplified_result = {
                        'model_id': model_id,
                        'generated_model_card': {}
                    }
                    return simplified_result
                    
        except Exception as e:
            logger.error(f"Exception generating card for {model_id}: {e}")
            
            # Return simplified result on exception
            simplified_result = {
                'model_id': model_id,
                'generated_model_card': {}
            }
            return simplified_result


    def read_and_filter_data(self, file_path: str) -> List[Dict]:
        """Read and filter generative AI data, retain complete original data"""
        logger.info(f"Reading and filtering generative AI models from: {file_path}")
        generative_models = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        # 添加调试信息
                        if line_num <= 3:  # 打印前3行的model_id
                            logger.info(f"Line {line_num} original model_id: '{data.get('model_id', 'NOT_FOUND')}'")
                        
                        # Only process generative AI models, but retain complete original data
                        if data.get('generative_ai_info', {}).get('is_generative_ai', False):
                            # Normalize data structure
                            normalized_data = self.normalize_data_structure(data)
                            
                            # 添加调试信息
                            if line_num <= 3:
                                logger.info(f"Line {line_num} normalized model_id: '{normalized_data.get('model_id', 'NOT_FOUND')}'")
                            
                            generative_models.append(normalized_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
            
        logger.info(f"Found {len(generative_models)} generative AI models")
        
        # Sort by popularity (downloads + likes)
        generative_models.sort(
            key=lambda x: x.get('downloads', 0) + x.get('likes', 0), 
            reverse=True
        )
        
        return generative_models


    async def process_batch_async(self, models_batch: List[Dict]) -> List[Dict]:
        """Async batch processing of models"""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single(model_data):
            async with semaphore:
                return await self.generate_model_card(model_data)
        
        tasks = [process_single(model) for model in models_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                model_id = models_batch[i].get('model_id', 'Unknown')
                logger.error(f"Task failed for {model_id}: {result}")
                # Return simplified result on failure
                simplified_result = {
                    'model_id': model_id,
                    'generated_model_card': {}
                }
                processed_results.append(simplified_result)
            else:
                processed_results.append(result)
        
        return processed_results


    def save_checkpoint(self, results: List[Dict], output_path: str, checkpoint_num: int):
        """Save checkpoint with only essential data"""
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


    async def process_all_models(self, input_file: str, output_file: str, batch_size: int = 50):
        """Process all generative AI models with progress bar and checkpoint resume"""
        # Read and filter data
        all_models = self.read_and_filter_data(input_file)
        
        # Check checkpoint
        processed_results = self.load_checkpoint(output_file)
        # 修复：正确处理空字符串的情况
        processed_ids = {r.get('model_id') for r in processed_results 
                        if r.get('model_id') is not None and r.get('model_id') != ''}
        
        # Filter already processed models
        remaining_models = [m for m in all_models 
                           if m.get('model_id') not in processed_ids]
        
        total_models = len(all_models)
        processed_count = len(processed_results)
        
        logger.info(f"Total generative AI models: {total_models}, Already processed: {processed_count}, Remaining: {len(remaining_models)}")
        
        # 添加调试信息
        if remaining_models:
            logger.info(f"First few remaining models: {[m.get('model_id', 'NO_ID') for m in remaining_models[:3]]}")
        
        # Process in batches with tqdm progress bar
        checkpoint_interval = 2  # Save checkpoint every 2 batches
        
        # Initialize tqdm with total models and already processed count
        pbar = None
        try:
            pbar = tqdm(total=total_models, initial=processed_count, unit="model", 
                       desc="Processing models", dynamic_ncols=True)
            
            for i in range(0, len(remaining_models), batch_size):
                batch = remaining_models[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{(len(remaining_models) + batch_size - 1) // batch_size}")
                
                # Process current batch
                batch_results = await self.process_batch_async(batch)
                processed_results.extend(batch_results)
                
                # Update progress bar
                pbar.update(len(batch))
                pbar.set_description(f"Processing batch {batch_num}")
                
                # Periodically save checkpoint
                if batch_num % checkpoint_interval == 0:
                    self.save_checkpoint(processed_results, output_file, batch_num)
                
                # Rate limiting
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user. Saving checkpoint...")
            # Save current progress on interrupt
            if processed_results:
                checkpoint_num = len(processed_results) // batch_size + 1
                self.save_checkpoint(processed_results, output_file, checkpoint_num)
            raise
        finally:
            # Close progress bar properly
            if pbar is not None:
                pbar.close()
        
        # Save final results as JSON format with only essential data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processing complete. Results saved to: {output_file}")
        
        # Generate detailed report
        self.generate_comprehensive_report(processed_results, output_file + '.report.json')
        
        return processed_results


    def generate_comprehensive_report(self, results: List[Dict], report_path: str):
        """Generate comprehensive report"""
        total = len(results)
        successful = len([r for r in results if r.get('generated_model_card', {}) != {}])
        failed = total - successful
        
        # Analyze confidence distribution
        confidence_stats = {
            'certain': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        # Analyze content type distribution
        content_stats = {
            'not_provided': 0,
            'not_applicable': 0,
            'with_content': 0
        }
        
        for result in results:
            card = result.get('generated_model_card', {})
            if card:
                for section in card.values():
                    if isinstance(section, dict):
                        for field in section.values():
                            if isinstance(field, dict):
                                if 'confidence' in field:
                                    conf = field['confidence'].lower()
                                    if conf in confidence_stats:
                                        confidence_stats[conf] += 1
                                
                                if 'content' in field:
                                    content = field['content'].lower()
                                    if 'not provided' in content:
                                        content_stats['not_provided'] += 1
                                    elif 'not applicable' in content:
                                        content_stats['not_applicable'] += 1
                                    else:
                                        content_stats['with_content'] += 1
        
        report = {
            'generation_summary': {
                'total_models': total,
                'successful': successful,
                'failed': failed,
                'success_rate': f"{(successful/total*100):.2f}%" if total > 0 else "0%",
                'timestamp': datetime.now().isoformat()
            },
            'confidence_distribution': confidence_stats,
            'content_distribution': content_stats,
            'failed_models': [
                {
                    'model_id': r.get('model_id', 'unknown')
                } 
                for r in results if r.get('generated_model_card', {}) == {}
            ],
            'framework_info': {
                'framework': 'Generative AI Model Card with Confidence Scoring (Simplified)',
                'saved_fields': ['model_id', 'generated_model_card'],
                'confidence_levels': ['certain', 'high', 'medium', 'low'],
                'content_indicators': ['Not provided', 'Not applicable', 'Available content']
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Comprehensive report generated: {report_path}")
        logger.info(f"Overall success rate: {report['generation_summary']['success_rate']} ({successful}/{total})")
        logger.info(f"Confidence distribution: {confidence_stats}")
        logger.info(f"Content distribution: {content_stats}")


async def main():
    """Main function"""
    API_KEY = os.getenv('OPENAI_API_KEY')
    if not API_KEY:
        logger.error("Please set OPENAI_API_KEY environment variable")
        return
    
    INPUT_FILE = "/mnt/shared_disk/datacards/dataset/processed/cleaned_model_cards_with_generative_ai.jsonl"
    OUTPUT_FILE = "/mnt/shared_disk/datacards/dataset/processed/generative_ai_model_cards.json"
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Use optimized generator
    async with GenerativeAIModelCardGenerator(API_KEY, max_workers=150) as generator:
        try:
            results = await generator.process_all_models(INPUT_FILE, OUTPUT_FILE, batch_size=30)
            logger.info("Generative AI Model Cards generation completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted. Progress saved in checkpoint files.")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
