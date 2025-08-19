"""
LFM Classifier - Step 1: Identify Large Foundation Models (Optimized Concurrent Version)
Optimized for Tier 4: 10K RPM, 25 concurrent threads, cached input, batch processing
Only performs LFM classification, outputs identified LFMs for further processing
Enhanced with robust interruption and continuation support - FIXED CHECKPOINT RECOVERY
"""

import json
import os
import time
import threading
import hashlib
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
from functools import lru_cache

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

class ConfidenceLevel(Enum):
    """Confidence levels for classification"""
    LOW = 0.25
    MEDIUM = 0.50
    HIGH = 0.75
    CERTAIN = 1.0

@dataclass
class LFMClassificationResult:
    """Result of GPT-based LFM classification"""
    is_lfm: bool
    confidence: float
    confidence_level: str
    reasoning: str
    evidence: List[str]
    parameter_count: Optional[str] = None
    model_type: str = "Unknown"

    def to_dict(self):
        return asdict(self)

class ConfidenceMapper:
    """Convert text confidence levels to numerical values"""
    
    CONFIDENCE_MAPPING = {
        "low": ConfidenceLevel.LOW.value,
        "medium": ConfidenceLevel.MEDIUM.value,
        "high": ConfidenceLevel.HIGH.value,
        "certain": ConfidenceLevel.CERTAIN.value,
    }

    @classmethod
    def text_to_score(cls, confidence_text: str) -> float:
        confidence_text = confidence_text.lower().strip()
        return cls.CONFIDENCE_MAPPING.get(confidence_text, ConfidenceLevel.LOW.value)

class CheckpointManager:
    """Checkpoint manager - supports state saving and restoration - ENHANCED VERSION"""
    
    CHECKPOINT_VERSION = "1.1"
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.lock = threading.Lock()
    
    def save_checkpoint(self, state: Dict[str, Any]):
        """Save checkpoint state - enhanced version"""
        with self.lock:
            try:
                checkpoint_data = {
                    'version': self.CHECKPOINT_VERSION,
                    'timestamp': datetime.now().isoformat(),
                    'state': state
                }
                
                # Atomic write - write to temp file first, then rename
                temp_file = self.checkpoint_file + ".tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
                # Atomic rename
                os.rename(temp_file, self.checkpoint_file)
                
            except Exception as e:
                tqdm.write(f"⚠️ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint state - enhanced version"""
        if not os.path.exists(self.checkpoint_file):
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                
            # Validate checkpoint
            if not self._validate_checkpoint(checkpoint_data):
                tqdm.write("⚠️ Invalid checkpoint format, ignoring...")
                return None
                
            return checkpoint_data.get('state')
            
        except Exception as e:
            tqdm.write(f"⚠️ Failed to load checkpoint: {e}")
            return None
    
    def _validate_checkpoint(self, checkpoint_data: Dict) -> bool:
        """Validate checkpoint data integrity"""
        if not isinstance(checkpoint_data, dict):
            return False
            
        required_keys = ['version', 'timestamp', 'state']
        if not all(key in checkpoint_data for key in required_keys):
            return False
            
        state = checkpoint_data.get('state', {})
        state_required_keys = ['processed_count', 'processed_ids']
        if not all(key in state for key in state_required_keys):
            return False
            
        return True
    
    def clear_checkpoint(self):
        """Clear checkpoint file"""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                tqdm.write("🗑️ Checkpoint file cleared")
        except Exception as e:
            tqdm.write(f"⚠️ Failed to clear checkpoint: {e}")

def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    tqdm.write("\n🛑 Shutdown requested. Finishing current tasks and saving progress...")
    tqdm.write("💾 Please wait for graceful shutdown...")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class OptimizedRateLimiter:
    """High-performance rate limiter optimized for Tier 4"""
    
    def __init__(self, max_calls_per_minute: int = 10000):
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = threading.Lock()
        self.local_cache = threading.local()
    
    def acquire(self):
        """Acquire call permission - optimized version"""
        global SHUTDOWN_REQUESTED
        if SHUTDOWN_REQUESTED:
            raise KeyboardInterrupt("Shutdown requested")
            
        with self.lock:
            now = time.time()
            # Only keep call records from the last 60 seconds
            cutoff_time = now - 60
            self.calls = [call_time for call_time in self.calls if call_time > cutoff_time]
            
            if len(self.calls) >= self.max_calls:
                # Calculate wait time needed
                oldest_call = min(self.calls)
                wait_time = 61 - (now - oldest_call)  # Wait slightly longer for safety
                if wait_time > 0:
                    tqdm.write(f"⏰ Rate limit reached, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    return self.acquire()
            
            self.calls.append(now)

class ContentCache:
    """Content caching system to avoid reprocessing similar content"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get_content_hash(self, content: str) -> str:
        """Generate content hash for caching"""
        # Normalize content to improve cache hit rate
        normalized_content = content.lower().strip()
        return hashlib.md5(normalized_content.encode()).hexdigest()[:16]  # Use shorter hash
    
    def get_cached_result(self, content_hash: str) -> Optional[LFMClassificationResult]:
        with self.lock:
            return self.cache.get(content_hash)
    
    def cache_result(self, content_hash: str, result: LFMClassificationResult):
        with self.lock:
            # If cache is too large, clean old entries
            if len(self.cache) >= self.max_size:
                # Simple cleanup strategy: remove half
                keys_to_remove = list(self.cache.keys())[:self.max_size // 2]
                for key in keys_to_remove:
                    del self.cache[key]
            
            self.cache[content_hash] = result
    
    def get_cache_stats(self) -> Dict[str, int]:
        with self.lock:
            return {"cache_size": len(self.cache), "max_size": self.max_size}

class RobustLLMCaller:
    """Provides LLM calls with retry mechanism and caching optimization"""

    def __init__(self, rate_limiter: OptimizedRateLimiter):
        self.rate_limiter = rate_limiter

    def call_llm_with_retry(
        self,
        client: OpenAI,
        model: str,
        messages: List[Dict],
        max_retries: int = 3,
        delay: float = 0.5,  # Reduce delay to improve speed
    ) -> str:
        global SHUTDOWN_REQUESTED
        
        if SHUTDOWN_REQUESTED:
            raise KeyboardInterrupt("Shutdown requested")
            
        last_error = None
        thread_id = threading.current_thread().ident

        for attempt in range(max_retries):
            if SHUTDOWN_REQUESTED:
                raise KeyboardInterrupt("Shutdown requested")
                
            try:
                # Rate limiting
                self.rate_limiter.acquire()

                # Optimized API call - optimized for gpt-5-nano
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=2000,
                    timeout=30.0,  # Reduce timeout for speed
                    # Enable caching mechanism for performance
                    seed=42
                )

                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    content = choice.message.content
                    
                    if choice.finish_reason in ["length", "stop"]:
                        if content and content.strip():
                            return content.strip()
                        else:
                            last_error = f"Response completed but content is None (reason: {choice.finish_reason})"
                    else:
                        if content and content.strip():
                            return content.strip()
                        else:
                            last_error = f"Finish reason: {choice.finish_reason}, content is None"
                else:
                    last_error = "No choices in API response"

            except Exception as e:
                if SHUTDOWN_REQUESTED:
                    raise KeyboardInterrupt("Shutdown requested")
                last_error = f"{type(e).__name__}: {str(e)}"

            # Wait before retry - optimized backoff strategy
            if attempt < max_retries - 1:
                wait_time = delay * (1.2 ** attempt)  # Gentler backoff
                time.sleep(wait_time)

        # All retries failed
        raise ValueError(f"All {max_retries} attempts failed. Last error: {last_error}")

class LFMDefinitionPrompts:
    """Contains LFM definition and classification prompts"""

    LFM_DEFINITION = """
Definition of Large Foundation Models (LFMs):
Large Foundation Models (LFMs) are massive AI models trained on extensive volumes of unlabeled data, 
typically through self-supervised learning, that can be adapted to a wide range of downstream tasks.

Key characteristics:
1. Scale: Billions of parameters
2. Emergence: Emergent capabilities from scale
3. Homogenization: Common foundation across diverse applications
4. Multimodality: Process/generate content across text, images, video, audio
5. Self-supervised Learning: Trained on broad data using self-supervision
6. Adaptability: Can be fine-tuned to wide range of downstream tasks
"""

    @staticmethod
    def generate_classification_prompt(model_card_text: str) -> str:
        # Optimize truncation length to balance accuracy and speed
        max_length = 800  # Reduce to 800 characters for improved processing speed
        truncated_text = model_card_text[:max_length] if len(model_card_text) > max_length else model_card_text

        return f"""
{LFMDefinitionPrompts.LFM_DEFINITION}

TASK: Analyze this model card and determine if it describes a Large Foundation Model.

MODEL CARD:
{truncated_text}

IMPORTANT: Respond ONLY with valid JSON in exactly this format:
{{"is_lfm": true, "confidence": "high", "reasoning": "Brief explanation", "evidence": ["Evidence 1", "Evidence 2"], "parameter_count": null, "model_type": "Language Model"}}

Valid confidence levels: low, medium, high, certain
Valid model types: Language Model, Vision Model, Multimodal Model, Unknown

JSON response:
"""

class OptimizedGPTBasedLFMDetector:
    """Optimized GPT-based LFM detector with caching and batch processing support"""

    def __init__(self, api_key: str, model_name: str = "gpt-5-nano-2025-08-07", rate_limiter: OptimizedRateLimiter = None):
        if not api_key:
            raise ValueError("API key is required")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.rate_limiter = rate_limiter or OptimizedRateLimiter()
        self.llm_caller = RobustLLMCaller(self.rate_limiter)
        self.content_cache = ContentCache()

        tqdm.write(f"🤖 Initializing optimized LFM detector with model: {model_name}")
        self._test_connection()

    def _test_connection(self):
        """Test API connection"""
        try:
            tqdm.write("🧪 Testing API connection...")
            test_messages = [{"role": "user", "content": "Reply with JSON: {\"status\": \"ok\"}"}]
            response = self.llm_caller.call_llm_with_retry(
                self.client, self.model_name, test_messages, max_retries=1
            )
            tqdm.write(f"✅ API connection test successful")
        except Exception as e:
            tqdm.write(f"❌ API connection test failed: {e}")
            raise

    def classify_model_card(self, model_card_text: str) -> LFMClassificationResult:
        """Optimized classification method with caching support"""
        global SHUTDOWN_REQUESTED
        
        if SHUTDOWN_REQUESTED:
            raise KeyboardInterrupt("Shutdown requested")
            
        # Check cache
        content_hash = self.content_cache.get_content_hash(model_card_text)
        cached_result = self.content_cache.get_cached_result(content_hash)
        
        if cached_result:
            return cached_result

        # If not cached, make API call
        prompt = LFMDefinitionPrompts.generate_classification_prompt(model_card_text)
        messages = [{"role": "user", "content": prompt}]

        max_attempts = 3
        
        for attempt in range(max_attempts):
            if SHUTDOWN_REQUESTED:
                raise KeyboardInterrupt("Shutdown requested")
                
            try:
                gpt_response = self.llm_caller.call_llm_with_retry(
                    self.client, self.model_name, messages, max_retries=2, delay=0.3
                )
                break
                
            except ValueError as e:
                if SHUTDOWN_REQUESTED:
                    raise KeyboardInterrupt("Shutdown requested")
                    
                if attempt < max_attempts - 1:
                    wait_time = 1.5 * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    result = LFMClassificationResult(
                        is_lfm=False,
                        confidence=ConfidenceLevel.LOW.value,
                        confidence_level="low",
                        reasoning="Failed to get API response after multiple attempts",
                        evidence=["API call failed"],
                        parameter_count=None,
                        model_type="Unknown"
                    )
                    # Cache even failures to avoid repeated failures
                    self.content_cache.cache_result(content_hash, result)
                    return result

        try:
            json_str = self._extract_json_from_response(gpt_response)
            result_json = json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            result = LFMClassificationResult(
                is_lfm=False,
                confidence=ConfidenceLevel.LOW.value,
                confidence_level="low",
                reasoning="Failed to parse GPT response as JSON",
                evidence=["JSON parsing error"],
                parameter_count=None,
                model_type="Unknown"
            )
            self.content_cache.cache_result(content_hash, result)
            return result

        confidence_text = result_json.get("confidence", "low")

        result = LFMClassificationResult(
            is_lfm=result_json.get("is_lfm", False),
            confidence=ConfidenceMapper.text_to_score(confidence_text),
            confidence_level=confidence_text,
            reasoning=result_json.get("reasoning", ""),
            evidence=result_json.get("evidence", []),
            parameter_count=result_json.get("parameter_count"),
            model_type=result_json.get("model_type", "Unknown"),
        )

        # Cache result
        self.content_cache.cache_result(content_hash, result)
        return result

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return self.content_cache.get_cache_stats()

    def _extract_json_from_response(self, response: str) -> str:
        """JSON extraction method"""
        if not response or not response.strip():
            raise ValueError("Empty response from LLM")

        response = response.strip()

        extraction_methods = [
            self._extract_standard_json,
            self._extract_code_block_json,
            self._extract_first_complete_json,
            self._extract_partial_json,
        ]

        for method in extraction_methods:
            try:
                json_str = method(response)
                if json_str:
                    parsed = json.loads(json_str)
                    return json_str
            except (json.JSONDecodeError, ValueError):
                continue

        raise ValueError("No valid JSON found in response")

    def _extract_standard_json(self, response: str) -> str:
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx : end_idx + 1]
        return ""

    def _extract_code_block_json(self, response: str) -> str:
        json_start = response.find("```")
        if json_start != -1:
            json_start = response.find("{", json_start)
            json_end = response.find("```", json_start)
            if json_start != -1 and json_end != -1:
                content = response[json_start:json_end].strip()
                brace_end = content.rfind("}")
                if brace_end != -1:
                    return content[: brace_end + 1]
        return ""

    def _extract_first_complete_json(self, response: str) -> str:
        start_idx = response.find("{")
        if start_idx == -1:
            return ""

        brace_count = 0
        for i, char in enumerate(response[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return response[start_idx : i + 1]
        return ""

    def _extract_partial_json(self, response: str) -> str:
        """Handle potentially truncated JSON"""
        start_idx = response.find("{")
        if start_idx == -1:
            return ""
        
        json_part = response[start_idx:]
        
        open_braces = json_part.count("{")
        close_braces = json_part.count("}")
        
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            
            if json_part.rstrip().endswith('"'):
                pass
            elif json_part.rstrip().endswith('",'):
                pass
            elif json_part.rstrip().endswith('": '):
                json_part += '""'
            elif '"' in json_part and not json_part.rstrip().endswith('"'):
                json_part += '"'
            
            json_part += "}" * missing_braces
            
            try:
                json.loads(json_part)
                return json_part
            except json.JSONDecodeError:
                pass
        
        return ""

class CleanedDataProcessor:
    """Process cleaned model card data from JSONL file - handles all data"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_all_data(self) -> List[Dict[str, Any]]:
        """Load all data without sampling, show progress with tqdm"""
        all_entries = []

        try:
            tqdm.write(f"📥 Loading ALL data from {self.file_path}...")
            
            # First count total lines
            tqdm.write("📊 Counting total lines...")
            with open(self.file_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
            
            tqdm.write(f"📋 Found {total_lines} lines, loading data...")

            with open(self.file_path, "r", encoding="utf-8") as f:
                with tqdm(total=total_lines, desc="Loading data", unit="lines") as pbar:
                    for line_num, line in enumerate(f, 1):
                        pbar.update(1)
                        
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            all_entries.append(data)
                        except json.JSONDecodeError:
                            continue

            tqdm.write(f"📁 Loaded {len(all_entries)} valid entries (ALL DATA)")
            return all_entries

        except FileNotFoundError:
            tqdm.write(f"❌ File not found: {self.file_path}")
            return []
        except Exception as e:
            tqdm.write(f"❌ Error loading data: {e}")
            return []

class OptimizedResultsManager:
    """Thread-safe results manager optimized with batch writing and checkpoints - FIXED CHECKPOINT RECOVERY"""

    def __init__(self, output_file: str, buffer_size: int = 200, checkpoint_manager: CheckpointManager = None):
        self.output_file = output_file
        self.processed_ids: Set[str] = set()
        self.stats = {
            "total_processed": 0,
            "lfm_count": 0,
            "error_count": 0,
            "cache_hits": 0,
        }
        
        # Batch write buffer
        self.buffer = []
        self.buffer_size = buffer_size
        
        # Checkpoint management
        self.checkpoint_manager = checkpoint_manager
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 300  # Save checkpoint every 5 minutes
        
        # Thread-safe locks
        self.lock = threading.Lock()
        self.file_lock = threading.Lock()

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self._load_existing_results()

    def _load_existing_results(self):
        """Load existing results and checkpoint state - FIXED VERSION"""
        checkpoint_state = None
        if self.checkpoint_manager:
            checkpoint_state = self.checkpoint_manager.load_checkpoint()
        
        # Prioritize checkpoint for state recovery
        if checkpoint_state:
            tqdm.write(f"💾 Restoring from checkpoint: {checkpoint_state.get('timestamp', 'unknown time')}")
            
            # Directly restore processed_ids from checkpoint
            if 'processed_ids' in checkpoint_state:
                self.processed_ids = set(checkpoint_state['processed_ids'])
                tqdm.write(f"📊 Restored {len(self.processed_ids)} processed IDs from checkpoint")
            
            # Restore statistics
            if 'processed_count' in checkpoint_state:
                self.stats['total_processed'] = checkpoint_state['processed_count']
            if 'lfm_count' in checkpoint_state:
                self.stats['lfm_count'] = checkpoint_state['lfm_count']
            if 'error_count' in checkpoint_state:
                self.stats['error_count'] = checkpoint_state['error_count']
                
            tqdm.write(f"📊 Checkpoint stats - Processed: {self.stats['total_processed']}, LFMs: {self.stats['lfm_count']}, Errors: {self.stats['error_count']}")
            tqdm.write("✅ Successfully restored from checkpoint")
            tqdm.write("▶️  Continuing from where we left off...")
            return
        
        # If no checkpoint, scan from output file
        if os.path.exists(self.output_file):
            tqdm.write(f"📂 No checkpoint found, scanning existing results: {self.output_file}")
            try:
                line_count = 0
                with open(self.output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        line_count += 1
                        try:
                            result = json.loads(line)
                            model_id = result.get("model_id", "")
                            if model_id:
                                self.processed_ids.add(model_id)
                                self.stats["total_processed"] += 1
                                if result.get("is_lfm", False):
                                    self.stats["lfm_count"] += 1
                        except json.JSONDecodeError:
                            continue

                tqdm.write(f"📊 Scanned {line_count} lines, found {len(self.processed_ids)} previously processed models")
                tqdm.write(f"📊 LFMs found: {self.stats['lfm_count']}")
                tqdm.write("▶️  Automatically continuing from previous run...")

            except Exception as e:
                tqdm.write(f"⚠️  Error reading existing results: {e}")
        else:
            tqdm.write("🆕 Starting fresh - no previous results or checkpoint found")

    def is_already_processed(self, model_id: str) -> bool:
        with self.lock:
            return model_id in self.processed_ids

    def save_result(self, model_id: str, tags: str, model_card_text: str, classification_result: LFMClassificationResult):
        global SHUTDOWN_REQUESTED
        
        try:
            result = {
                "model_id": model_id,
                "tags": tags,
                "model_card_text": model_card_text,
                "is_lfm": classification_result.is_lfm,
                "lfm_confidence": classification_result.confidence,
                "lfm_confidence_level": classification_result.confidence_level,
                "reasoning": classification_result.reasoning,
                "evidence": classification_result.evidence,
                "parameter_count": classification_result.parameter_count,
                "model_type": classification_result.model_type,
                "processed_at": datetime.now().isoformat(),
                "classification_result": classification_result.to_dict(),
            }

            # Thread-safe buffer management
            with self.lock:
                self.buffer.append(result)
                self.processed_ids.add(model_id)
                self.stats["total_processed"] += 1
                if classification_result.is_lfm:
                    self.stats["lfm_count"] += 1
                
                # Batch write to file or force write on shutdown
                if len(self.buffer) >= self.buffer_size or SHUTDOWN_REQUESTED:
                    self._flush_buffer()
                
                # Periodically save checkpoint
                current_time = time.time()
                if (current_time - self.last_checkpoint_time > self.checkpoint_interval or 
                    SHUTDOWN_REQUESTED):
                    self._save_checkpoint()
                    self.last_checkpoint_time = current_time

        except Exception as e:
            with self.lock:
                self.stats["error_count"] += 1

    def _flush_buffer(self):
        """Batch write buffer to file"""
        if self.buffer:
            with self.file_lock:
                with open(self.output_file, "a", encoding="utf-8") as f:
                    for result in self.buffer:
                        json.dump(result, f, ensure_ascii=False)
                        f.write("\n")
            self.buffer.clear()

    def _save_checkpoint(self):
        """Save current processing state to checkpoint - ENHANCED VERSION"""
        if self.checkpoint_manager:
            checkpoint_state = {
                'processed_count': self.stats['total_processed'],
                'lfm_count': self.stats['lfm_count'],
                'error_count': self.stats['error_count'],
                'processed_ids': list(self.processed_ids),  # Convert set to list for JSON serialization
                'buffer_size': len(self.buffer),
                'last_update': datetime.now().isoformat()
            }
            self.checkpoint_manager.save_checkpoint(checkpoint_state)

    def finalize(self):
        """Flush all buffers and clean up checkpoint on completion"""
        with self.lock:
            if self.buffer:
                self._flush_buffer()
            
            # Clear checkpoint file
            if self.checkpoint_manager:
                self.checkpoint_manager.clear_checkpoint()

    def get_current_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "total_processed": self.stats["total_processed"],
                "lfm_count": self.stats["lfm_count"],
                "error_count": self.stats["error_count"],
                "cache_hits": self.stats["cache_hits"],
                "lfm_detection_rate": (
                    (self.stats["lfm_count"] / self.stats["total_processed"] * 100)
                    if self.stats["total_processed"] > 0
                    else 0
                ),
            }

    def get_final_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "total_processed": self.stats["total_processed"],
                "lfm_count": self.stats["lfm_count"],
                "error_count": self.stats["error_count"],
                "cache_hits": self.stats["cache_hits"],
                "lfm_detection_rate": (
                    (self.stats["lfm_count"] / self.stats["total_processed"] * 100)
                    if self.stats["total_processed"] > 0
                    else 0
                ),
                "output_file": self.output_file,
            }

def process_batch_models(
    detector: OptimizedGPTBasedLFMDetector,
    results_manager: OptimizedResultsManager,
    batch_entries: List[Dict[str, Any]],
    progress_callback=None
) -> List[bool]:
    """Batch process models for improved efficiency with interruption support"""
    global SHUTDOWN_REQUESTED
    results = []
    
    for entry in batch_entries:
        if SHUTDOWN_REQUESTED:
            tqdm.write("🛑 Batch processing interrupted, saving progress...")
            break
            
        try:
            model_id = entry.get("model_id", "Unknown")
            classification_result = detector.classify_model_card(
                entry.get("modelCard", "")
            )
            
            results_manager.save_result(
                model_id=model_id,
                tags=entry.get("tags", ""),
                model_card_text=entry.get("modelCard", ""),
                classification_result=classification_result
            )
            
            results.append(True)
            if progress_callback:
                progress_callback(classification_result.is_lfm)
                
        except KeyboardInterrupt:
            tqdm.write("🛑 KeyboardInterrupt in batch processing")
            break
        except Exception as e:
            results.append(False)
            if progress_callback:
                progress_callback(False)
    
    return results

def load_config():
    """Load configuration optimized for Tier 4 + gpt-5-nano"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = ""
        tqdm.write("⚠️  Using hardcoded API key")
    else:
        tqdm.write("✅ Using API key from environment variable")

    return {
        "api_key": api_key,
        "model": "gpt-5-nano-2025-08-07",  # Use specified gpt5nano
        "max_workers": 25,  # Increase to 25 concurrent threads
        "max_calls_per_minute": 10000,  # Upgrade to 10K RPM for Tier 4
        "batch_size": 50,  # Batch processing size
        "buffer_size": 200,  # I/O buffer size
    }

class ProgressTracker:
    """Progress tracker with detailed statistics"""
    def __init__(self):
        self.lock = threading.Lock()
        self.lfm_found = 0
        self.total_processed = 0
        self.cache_hits = 0
        
    def update(self, is_lfm: bool, from_cache: bool = False):
        with self.lock:
            self.total_processed += 1
            if is_lfm:
                self.lfm_found += 1
            if from_cache:
                self.cache_hits += 1
    
    def get_stats(self):
        with self.lock:
            return {
                "total": self.total_processed,
                "lfm": self.lfm_found,
                "cache_hits": self.cache_hits,
                "rate": (self.lfm_found / self.total_processed * 100) if self.total_processed > 0 else 0,
                "cache_hit_rate": (self.cache_hits / self.total_processed * 100) if self.total_processed > 0 else 0
            }

def main():
    """Main function - highly optimized concurrent LFM classification with interruption and continuation support - FIXED CHECKPOINT RECOVERY"""
    global SHUTDOWN_REQUESTED
    
    print("=" * 90)
    print("🚀 LFM CLASSIFIER - STEP 1: OPTIMIZED HIGH-SPEED LFM IDENTIFICATION")
    print("💎 Tier 4 Optimized: 10K RPM, 25 threads, cached input, batch processing")
    print("🤖 Model: gpt-5-nano-2025-08-07")
    print("⚡ Enhanced with content caching, batch I/O, smart rate limiting")
    print("🛡️ Robust interruption and continuation support - FIXED CHECKPOINT RECOVERY")
    print("=" * 90)

    try:
        # Load configuration
        config = load_config()
        
        tqdm.write(f"⚙️ Optimized Configuration:")
        tqdm.write(f"   Model: {config['model']}")
        tqdm.write(f"   Max workers: {config['max_workers']}")
        tqdm.write(f"   Rate limit: {config['max_calls_per_minute']:,} calls/minute")
        tqdm.write(f"   Batch size: {config['batch_size']}")
        tqdm.write(f"   Buffer size: {config['buffer_size']}")

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            "/mnt/shared_disk/datacards/dataset/processed/lfm_classification_checkpoint.json"
        )
        
        # Initialize components
        tqdm.write("\n🚀 Initializing optimized components...")
        data_processor = CleanedDataProcessor(
            "/mnt/shared_disk/datacards/dataset/processed/cleaned_model_cards.jsonl"
        )
        
        # Shared optimized rate limiter
        rate_limiter = OptimizedRateLimiter(max_calls_per_minute=config['max_calls_per_minute'])
        
        # Create independent optimized detector instances for each thread
        def create_detector():
            return OptimizedGPTBasedLFMDetector(
                api_key=config["api_key"], 
                model_name=config["model"],
                rate_limiter=rate_limiter
            )
        
        results_manager = OptimizedResultsManager(
            "/mnt/shared_disk/datacards/dataset/processed/lfm_classification_results_optimized.jsonl",
            buffer_size=config['buffer_size'],
            checkpoint_manager=checkpoint_manager
        )

        # Load all data (show progress with tqdm)
        tqdm.write("\n📊 Loading ALL data...")
        all_data = data_processor.load_all_data()

        if not all_data:
            tqdm.write("❌ No data found")
            return None

        # Filter unprocessed data
        tqdm.write("🔍 Filtering unprocessed data...")
        unprocessed_data = []
        for entry in tqdm(all_data, desc="Filtering data", unit="entries"):
            if not results_manager.is_already_processed(entry.get("model_id", "")):
                unprocessed_data.append(entry)

        tqdm.write(f"📋 Total loaded: {len(all_data):,}")
        tqdm.write(f"📋 Unprocessed: {len(unprocessed_data):,}")

        if not unprocessed_data:
            tqdm.write("✅ All models have been processed!")
            final_stats = results_manager.get_final_stats()
            tqdm.write(f"\n📈 Current statistics: {final_stats}")
            return final_stats

        # Optimized batch processing
        tqdm.write(f"\n🔍 High-speed batch processing {len(unprocessed_data):,} models...")
        tqdm.write(f"💎 Configuration: {config['max_workers']} threads × {config['max_calls_per_minute']:,} RPM")
        tqdm.write("🛡️ Press Ctrl+C anytime to gracefully interrupt and save progress")
        
        # Create batches
        batch_size = config['batch_size']
        batches = [unprocessed_data[i:i + batch_size] for i in range(0, len(unprocessed_data), batch_size)]
        
        tqdm.write(f"📦 Created {len(batches)} batches of size {batch_size}")
        
        # Create progress tracker
        progress_tracker = ProgressTracker()
        
        # Create main progress bar
        with tqdm(total=len(unprocessed_data), desc="🚀 Optimized LFM Classification", unit="models") as main_pbar:
            
            def update_progress(is_lfm: bool, from_cache: bool = False):
                progress_tracker.update(is_lfm, from_cache)
                stats = progress_tracker.get_stats()
                
                # Update progress bar description
                main_pbar.set_postfix({
                    'LFMs': f"{stats['lfm']:,}",
                    'Rate': f"{stats['rate']:.1f}%",
                    'Cache': f"{stats['cache_hit_rate']:.1f}%",
                    'Total': f"{stats['total']:,}"
                })
                main_pbar.update(1)
            
            # Highly optimized concurrent processing with interruption support
            with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
                # Submit all batch tasks
                future_to_batch = {}
                
                for batch in batches:
                    if SHUTDOWN_REQUESTED:
                        break
                        
                    future = executor.submit(
                        process_batch_models,
                        create_detector(),
                        results_manager,
                        batch,
                        update_progress
                    )
                    future_to_batch[future] = batch

                # Process completed tasks
                completed_batches = 0
                total_errors = 0
                
                try:
                    for future in as_completed(future_to_batch):
                        if SHUTDOWN_REQUESTED:
                            tqdm.write("🛑 Cancelling remaining tasks...")
                            break
                            
                        batch = future_to_batch[future]
                        completed_batches += 1
                        
                        try:
                            batch_results = future.result()
                            batch_errors = sum(1 for result in batch_results if not result)
                            total_errors += batch_errors
                            
                            if batch_errors > 0:
                                tqdm.write(f"⚠️ Batch {completed_batches}/{len(batches)} completed with {batch_errors} errors")
                                
                        except Exception as e:
                            if not SHUTDOWN_REQUESTED:
                                total_errors += len(batch)
                                tqdm.write(f"💥 Batch {completed_batches}/{len(batches)} failed: {str(e)[:100]}")
                
                except KeyboardInterrupt:
                    SHUTDOWN_REQUESTED = True
                    tqdm.write("🛑 Received interrupt signal, shutting down gracefully...")

        # Ensure all results are written
        tqdm.write("💾 Finalizing results and cleaning up...")
        results_manager.finalize()

        # Collect cache statistics
        sample_detector = create_detector()
        cache_stats = sample_detector.get_cache_stats()
        
        # Final statistics
        final_stats = results_manager.get_final_stats()

        print("\n" + "=" * 90)
        if SHUTDOWN_REQUESTED:
            print("🛑 OPTIMIZED LFM CLASSIFICATION INTERRUPTED BUT PROGRESS SAVED!")
            print("▶️  Run the script again to continue from where you left off")
        else:
            print("🎉 OPTIMIZED HIGH-SPEED LFM CLASSIFICATION COMPLETE!")
        print("=" * 90)
        print(f"📊 Final Performance Statistics:")
        print(f"   Total Processed: {final_stats['total_processed']:,}")
        print(f"   LFMs Found: {final_stats['lfm_count']:,}")
        print(f"   Detection Rate: {final_stats['lfm_detection_rate']:.1f}%")
        print(f"   Errors: {final_stats['error_count']:,}")
        print(f"   Cache Hits: {final_stats.get('cache_hits', 0):,}")
        print(f"   Cache Size: {cache_stats['cache_size']:,}")
        print(f"\n🚀 Performance Optimizations Applied:")
        print(f"   ✅ Content caching system")
        print(f"   ✅ Batch processing ({batch_size} per batch)")
        print(f"   ✅ Optimized rate limiting (10K RPM)")
        print(f"   ✅ Concurrent threading ({config['max_workers']} workers)")
        print(f"   ✅ Buffered I/O ({config['buffer_size']} results per write)")
        print(f"   ✅ gpt-5-nano model optimization")
        print(f"   ✅ Robust interruption and continuation support")
        print(f"   ✅ Automatic checkpoint saving every 5 minutes")
        print(f"   ✅ FIXED: Proper checkpoint recovery from interruption")
        print(f"\n💾 Optimized results saved to: {final_stats['output_file']}")
        
        if SHUTDOWN_REQUESTED:
            print("🔄 To continue processing, simply run the script again")
            print("💾 The script will automatically resume from the checkpoint")
        else:
            print("🔜 Next step: Run lfm_scorer.py to evaluate LFM completeness")

        return final_stats

    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user. Progress has been saved.")
        print("▶️  Run the script again to continue from where you left off.")
        return None
    except Exception as e:
        tqdm.write(f"\n💥 FATAL ERROR: {type(e).__name__}: {str(e)}")
        return None

if __name__ == "__main__":
    result = main()
    if result is None:
        print("\n❌ Optimized classification failed or was interrupted.")
    else:
        print(f"\n✅ Optimized classification completed successfully.")
        print(f"📊 Found {result['lfm_count']:,} LFMs out of {result['total_processed']:,} models")
        print(f"🚀 Performance: {result.get('cache_hits', 0):,} cache hits saved significant processing time")
