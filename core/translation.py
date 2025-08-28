import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from itertools import chain
import xbmc

from .srt import parse_srt, group_blocks, write_srt
from .prompt import build_prompt, extract_translations
from .backoff import get_provider_name_from_fn

# Provider-specific batch sizing configuration
PROVIDER_BATCH_CONFIG = {
    "OpenAI": {
        "max_batch_size": 20,
        "max_content_length": 8000,
        "max_parallel": 5
    },
    "Gemini": {
        "max_batch_size": 15,
        "max_content_length": 10000,
        "max_parallel": 3
    },
    "OpenRouter": {
        "max_batch_size": 10,
        "max_content_length": 6000,
        "max_parallel": 3
    },
    "default": {
        "max_batch_size": 15,
        "max_content_length": 8000,
        "max_parallel": 3
    }
}

def get_provider_batch_config(call_fn):
    """Get provider-specific batch configuration"""
    provider_name = get_provider_name_from_fn(call_fn)
    return PROVIDER_BATCH_CONFIG.get(provider_name, PROVIDER_BATCH_CONFIG["default"])

def calculate_dynamic_batch_size(batch, max_content_length):
    """Calculate appropriate batch size based on content length"""
    if not batch:
        return 1
        
    # Calculate total content length for the batch
    total_length = sum(len("\n".join(b["lines"])) for _, b in batch)
    
    # If total content is already small, we can process more items
    if total_length < max_content_length // 2:
        return min(len(batch), max(1, int(max_content_length / (total_length / len(batch) if len(batch) > 0 else 1))))
    
    # Otherwise, we need to reduce batch size to fit within limits
    current_size = len(batch)
    while current_size > 0:
        # Estimate content length for this batch size
        estimated_length = (total_length / len(batch)) * current_size
        if estimated_length <= max_content_length:
            return current_size
        current_size -= 1
    
    return 1  # At least process one item

def translate_batch(batch, lang, model, api_key, call_fn, max_retries=3):
    """Translate a batch with retry logic and better error handling"""
    indexed_texts = [(i, "\n".join(b["lines"])) for i, b in batch]
    prompt = build_prompt(indexed_texts, lang)
    
    # Try to translate with retries
    for attempt in range(max_retries):
        try:
            response = call_fn(prompt, model, api_key)
            translations = extract_translations(response)  # musi zwracać dict: i -> text
            
            missing = [i for i, _ in batch if i not in translations]
            if missing:
                xbmc.log(f"[TRANSLATION] Missing translations for indices: {missing}", xbmc.LOGWARNING)
                xbmc.log("=== Prompt ===\n" + prompt, xbmc.LOGDEBUG)
                xbmc.log("=== Response ===\n" + response, xbmc.LOGDEBUG)
                
                # If we have some translations, we can still return them
                partial_results = [(i, translations[i]) for i, _ in batch if i in translations]
                
                # If this is the last attempt, return what we have
                if attempt == max_retries - 1:
                    xbmc.log(f"[TRANSLATION] Giving up on missing translations after {max_retries} attempts", xbmc.LOGERROR)
                    return partial_results
                
                # Wait before retrying
                time.sleep(2 ** attempt)
                continue
                
            return [(i, translations[i]) for i, _ in batch if i in translations]
                
            return [(i, translations[i]) for i, _ in batch if i in translations]
            
        except Exception as e:
            xbmc.log(f"[TRANSLATION] Batch translation failed (attempt {attempt + 1}/{max_retries}): {str(e)}", xbmc.LOGERROR)
            xbmc.log(traceback.format_exc(), xbmc.LOGERROR)
            
            # If this is the last attempt, re-raise the exception
            if attempt == max_retries - 1:
                raise
                
            # Wait before retrying
            time.sleep(2 ** attempt)
    
    # This should never be reached due to the re-raise above
    return []


from itertools import chain

def execute_batch_group(group, lang, model, api_key, call_fn, max_workers=None):
    """Execute a group of batches with optimized thread pool usage"""
    # Use provided max_workers or default to number of batches
    if max_workers is None:
        max_workers = min(len(group), 10)  # Cap at 10 workers to prevent resource exhaustion
    
    results = []
    failed_batches = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(translate_batch, batch, lang, model, api_key, call_fn): batch
            for batch in group
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_result = future.result()
                results.extend(batch_result)
            except Exception as e:
                xbmc.log(f"[TRANSLATION] Batch failed: {str(e)}", xbmc.LOGERROR)
                xbmc.log(traceback.format_exc(), xbmc.LOGERROR)
                # Store failed batch for potential retry
                failed_batches.append((batch, e))
    
    # Handle failed batches with individual item processing
    if failed_batches:
        xbmc.log(f"[TRANSLATION] Processing {len(failed_batches)} failed batches individually", xbmc.LOGWARNING)
        recovered_items = 0
        total_items = sum(len(batch) for batch, _ in failed_batches)
        
        for batch, error in failed_batches:
            # Try processing each item individually
            for item in batch:
                try:
                    single_result = translate_batch([item], lang, model, api_key, call_fn, max_retries=2)
                    results.extend(single_result)
                    recovered_items += len(single_result)
                except Exception as e:
                    xbmc.log(f"[TRANSLATION] Failed to translate item {item[0]}: {str(e)}", xbmc.LOGERROR)
                    # We continue processing other items even if one fails
        
        xbmc.log(f"[TRANSLATION] Recovered {recovered_items}/{total_items} items from failed batches", xbmc.LOGINFO)
    
    return results


def translate_in_batches(batches, lang, model, api_key, call_fn, parallel, report_progress=None, check_cancelled=None):
    """Translate batches with improved error handling and progress reporting"""
    results = []
    batch_iter = iter(batches)
    total = len(batches)
    done = 0
    
    # Get provider-specific configuration
    provider_config = get_provider_batch_config(call_fn)
    max_parallel = min(parallel, provider_config["max_parallel"])
    
    def next_group():
        return list(islice(batch_iter, max_parallel))
    
    group = next_group()
    group_count = 0
    
    while group:
        if check_cancelled and check_cancelled():
            raise Exception("Translation interrupted by client")
        
        group_count += 1
        xbmc.log(f"[TRANSLATION] Processing group {group_count} with {len(group)} batches", xbmc.LOGINFO)
        
        try:
            group_results = execute_batch_group(group, lang, model, api_key, call_fn)
            results.extend(group_results)
            
            done += len(group)
            if report_progress:
                report_progress(done, total)
                
            xbmc.log(f"[TRANSLATION] Completed group {group_count}: {len(group_results)} translations", xbmc.LOGINFO)
            
        except Exception as e:
            xbmc.log(f"[TRANSLATION] Group {group_count} failed: {str(e)}", xbmc.LOGERROR)
            xbmc.log(traceback.format_exc(), xbmc.LOGERROR)
            # Continue with next group instead of failing completely
            done += len(group)
            if report_progress:
                report_progress(done, total)
            # Still add some result reporting for failed groups
            xbmc.log(f"[TRANSLATION] Continuing despite failure in group {group_count}", xbmc.LOGWARNING)
        
        group = next_group()
    
    return results


def merge_translations(blocks, translated_pairs):
    """Merge translations with better error handling"""
    translated_map = dict(translated_pairs)
    merged = []
    
    success_count = 0
    failure_count = 0
    
    for i, block in enumerate(blocks):
        if i in translated_map:
            try:
                merged.append({**block, "lines": translated_map[i].split("\n")})
                success_count += 1
            except Exception as e:
                xbmc.log(f"[TRANSLATION] Failed to merge translation for block {i}: {str(e)}", xbmc.LOGERROR)
                # Keep original block if merge fails
                merged.append(block)
                failure_count += 1
        else:
            xbmc.log(f"[TRANSLATION] No translation found for block {i}", xbmc.LOGWARNING)
            # Keep original block if no translation
            merged.append(block)
            failure_count += 1
    
    xbmc.log(f"[TRANSLATION] Merge results: {success_count} successful, {failure_count} failed", xbmc.LOGINFO)
    return merged


def create_dynamic_batches(enumerated_blocks, call_fn):
    """Create batches dynamically based on content length and provider capabilities"""
    provider_config = get_provider_batch_config(call_fn)
    max_batch_size = provider_config["max_batch_size"]
    max_content_length = provider_config["max_content_length"]
    
    batches = []
    current_batch = []
    current_content_length = 0
    
    for i, block in enumerated_blocks:
        block_content_length = len("\n".join(block["lines"]))
        
        # If adding this block would exceed limits, start a new batch
        if (current_batch and
            (len(current_batch) >= max_batch_size or
             current_content_length + block_content_length > max_content_length)):
            batches.append(current_batch)
            current_batch = []
            current_content_length = 0
        
        # Add block to current batch
        current_batch.append((i, block))
        current_content_length += block_content_length
    
    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
    
    xbmc.log(f"[TRANSLATION] Created {len(batches)} dynamic batches", xbmc.LOGINFO)
    return batches


def translate_subtitles(
    path,
    api_key,
    lang,
    model,
    call_fn,
    report_progress=None,
    check_cancelled=None,
    parallel=3
):
    """Translate subtitles with all optimizations"""
    xbmc.log(f"[TRANSLATION] Starting translation of {path}", xbmc.LOGINFO)
    start_time = time.time()
    
    blocks = parse_srt(path)
    xbmc.log(f"[TRANSLATION] Parsed {len(blocks)} blocks from SRT", xbmc.LOGINFO)
    
    # Create dynamic batches based on content and provider capabilities
    enumerated_blocks = list(enumerate(blocks))
    batches = create_dynamic_batches(enumerated_blocks, call_fn)
    
    xbmc.log(f"[TRANSLATION] Created {len(batches)} batches for translation", xbmc.LOGINFO)
    
    # Translate batches
    translated_pairs = translate_in_batches(
        batches, lang, model, api_key, call_fn, parallel,
        report_progress, check_cancelled
    )
    
    xbmc.log(f"[TRANSLATION] Completed translation of {len(translated_pairs)} blocks", xbmc.LOGINFO)
    
    # Merge translations
    merged = merge_translations(blocks, translated_pairs)
    
    # Calculate success rate
    success_rate = len(translated_pairs) / len(blocks) * 100 if blocks else 0
    xbmc.log(f"[TRANSLATION] Translation success rate: {success_rate:.2f}%", xbmc.LOGINFO)
    
    # Write result
    new_path = path.replace(".srt", f".{lang.lower()}.translated.srt")
    write_result = write_srt(merged, new_path)
    
    end_time = time.time()
    duration = end_time - start_time
    xbmc.log(f"[TRANSLATION] Translation completed in {duration:.2f} seconds", xbmc.LOGINFO)
    
    # Report final statistics
    if report_progress:
        report_progress(len(batches), len(batches))  # Mark as complete
    
    return write_result or new_path
