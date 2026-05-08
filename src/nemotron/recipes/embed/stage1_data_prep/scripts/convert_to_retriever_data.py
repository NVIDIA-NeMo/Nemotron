#!/usr/bin/env python3
"""
Convert SDG generated output to retriever data formats.

This script converts the generated JSON files from the SDG pipeline into:
1. Training format: train.json (retriever training format)
2. Validation format: val.json (same format as training)
3. Test/Evaluation format: BEIR-compatible format (corpus.jsonl, queries.jsonl, qrels/test.tsv)
4. Corpus: parquet + metadata (shared across all splits)

Supports both:
- A folder of batch files (generated_batch*.json)
- A single merged JSON file

Supports splitting data into train/val/test sets with configurable ratios (default: 0.8/0.1/0.1).

Usage:
    # Full train/val/test split (default 80/10/10)
    python convert_to_retriever_data.py ./generated_output --corpus-id my_corpus

    # Custom split ratios
    python convert_to_retriever_data.py ./generated_output --corpus-id my_corpus --train-ratio 0.8 --val-ratio 0.1

    # Evaluation only (100% test, BEIR format output)
    python convert_to_retriever_data.py ./generated_output --corpus-id my_corpus --eval-only

    # With quality threshold
    python convert_to_retriever_data.py ./generated_output --corpus-id my_corpus --quality-threshold 8.0
"""

import json
import os
import argparse
import hashlib
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import pandas as pd
import numpy as np


def filter_qa_pairs_by_quality(
    generated_df: pd.DataFrame,
    quality_threshold: float = 7.0
) -> tuple[pd.DataFrame, list[dict]]:
    """Filter deduplicated_qa_pairs based on qa_evaluations scores.

    This function filters QA pairs from the generated data based on quality scores
    from the qa_evaluations column. Each QA pair's overall score is compared against
    the threshold, and only pairs meeting the threshold are returned.

    Files with data integrity issues (mismatched QA pairs and evaluations) are skipped.

    Args:
        generated_df: DataFrame with 'deduplicated_qa_pairs', 'qa_evaluations', and 'file_name' columns
        quality_threshold: Minimum overall quality score for QA pairs (default: 7.0)

    Returns:
        Tuple of:
        - DataFrame containing filtered QA pairs with columns from the original QA pairs,
          plus 'file_name' and 'quality_score' columns added.
        - List of dicts with 'file_name' and 'reason' for each skipped file.
    """
    print(f"Filtering QA pairs based on quality threshold: {quality_threshold}")

    total_pairs = 0
    filtered_pairs = 0
    all_filtered_qa_pairs = []
    skipped_files = []  # Track files skipped due to data integrity issues

    for _, row in generated_df.iterrows():
        file_name = row.get('file_name', 'unknown')

        # Get deduplicated QA pairs
        deduplicated_qa_pairs = row.get('deduplicated_qa_pairs')

        # Get QA evaluations
        qa_evaluations = row.get('qa_evaluations')

        # Handle case where deduplicated_qa_pairs might be None
        if deduplicated_qa_pairs is None:
            print(f"Warning: Skipping {file_name} - deduplicated_qa_pairs is None")
            continue

        # Convert numpy array to list if needed (parquet stores as numpy arrays)
        if isinstance(deduplicated_qa_pairs, np.ndarray):
            deduplicated_qa_pairs = deduplicated_qa_pairs.tolist()

        # Handle case where pairs is not a list or is empty
        if not isinstance(deduplicated_qa_pairs, (list, np.ndarray)) or len(deduplicated_qa_pairs) == 0:
            print(f"Warning: Skipping {file_name} - no valid deduplicated pairs found")
            continue

        # Extract evaluation scores
        evaluation_scores = []
        if qa_evaluations is not None:
            if isinstance(qa_evaluations, dict):
                evaluations_list = qa_evaluations.get('evaluations', [])
            else:
                evaluations_list = getattr(qa_evaluations, 'evaluations', [])

            # Convert numpy array to list if needed
            if isinstance(evaluations_list, np.ndarray):
                evaluations_list = evaluations_list.tolist()

            # Extract overall scores
            for eval_item in evaluations_list:
                if isinstance(eval_item, dict):
                    overall = eval_item.get('overall', {})
                    if isinstance(overall, dict):
                        score = overall.get('score', 0)
                    else:
                        score = getattr(overall, 'score', 0)
                else:
                    overall = getattr(eval_item, 'overall', None)
                    if overall is not None:
                        if isinstance(overall, dict):
                            score = overall.get('score', 0)
                        else:
                            score = getattr(overall, 'score', 0)
                    else:
                        score = 0
                evaluation_scores.append(score)

        # Validate that deduplicated_qa_pairs and qa_evaluations have the same length
        if len(evaluation_scores) != len(deduplicated_qa_pairs):
            reason = (
                f"deduplicated_qa_pairs has {len(deduplicated_qa_pairs)} items "
                f"but qa_evaluations has {len(evaluation_scores)} items"
            )
            print(f"Warning: Skipping {file_name} - data integrity error: {reason}")
            skipped_files.append({'file_name': file_name, 'reason': reason})
            continue

        # Filter QA pairs based on quality threshold
        for pair_idx, qa_pair in enumerate(deduplicated_qa_pairs):
            total_pairs += 1

            # Get quality score for this pair
            quality_score = evaluation_scores[pair_idx] if pair_idx < len(evaluation_scores) else 0

            # Filter based on quality threshold
            if quality_score >= quality_threshold:
                # Add file_name and quality_score to the QA pair
                if isinstance(qa_pair, dict):
                    qa_pair_with_metadata = qa_pair.copy()
                else:
                    # Handle Pydantic model objects by converting to dict
                    qa_pair_with_metadata = {
                        'question': getattr(qa_pair, 'question', ''),
                        'answer': getattr(qa_pair, 'answer', ''),
                        'query_type': getattr(qa_pair, 'query_type', ''),
                        'reasoning_type': getattr(qa_pair, 'reasoning_type', ''),
                        'question_complexity': getattr(qa_pair, 'question_complexity', 0),
                        'segment_ids': getattr(qa_pair, 'segment_ids', []),
                        'hop_count': getattr(qa_pair, 'hop_count', 1),
                        'hop_contexts': getattr(qa_pair, 'hop_contexts', []),
                    }
                qa_pair_with_metadata['file_name'] = file_name
                qa_pair_with_metadata['quality_score'] = quality_score
                all_filtered_qa_pairs.append(qa_pair_with_metadata)
            else:
                filtered_pairs += 1

    # Create dataframe from filtered QA pairs
    filtered_df = pd.DataFrame(all_filtered_qa_pairs)

    # Print statistics
    print(f"\nQuality Filtering Results:")
    print(f"  Total QA pairs: {total_pairs}")
    print(f"  Filtered out (score < {quality_threshold}): {filtered_pairs}")
    print(f"  Remaining high-quality pairs: {len(filtered_df)}")
    print(f"  Files skipped due to data issues: {len(skipped_files)}")
    if total_pairs > 0:
        print(f"  Retention rate: {len(filtered_df)/total_pairs*100:.1f}%")
    else:
        print("  Retention rate: 0%")

    return filtered_df, skipped_files


def filter_mismatched_records(
    records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filter out records where qa_evaluations.evaluations and deduplicated_qa_pairs
    have different sizes.
    
    Returns:
        - Filtered records
        - Count of dropped records
    """
    filtered = []
    dropped_count = 0
    
    for record in records:
        # Get sizes, defaulting to 0 if missing
        qa_evals = record.get("qa_evaluations", {}).get("evaluations", [])
        dedup_pairs = record.get("deduplicated_qa_pairs", [])
        
        eval_count = len(qa_evals)
        pairs_count = len(dedup_pairs)
        
        if eval_count == pairs_count:
            filtered.append(record)
        else:
            dropped_count += 1
            file_name = record.get("file_name", "unknown")
            # Handle both list and string file_name for display
            display_name = file_name if isinstance(file_name, str) else ", ".join(file_name) if file_name else "unknown"
            print(f"  Dropping record '{display_name}': "
                  f"qa_evaluations={eval_count}, deduplicated_qa_pairs={pairs_count}")
    
    return filtered, dropped_count


def get_corpus_id(text: str) -> str:
    """
    Generate hash-based corpus ID from text.
    Uses SHA256 hash with 16-character prefix.
    
    Args:
        text: Text content to hash
        
    Returns:
        Corpus ID in format 'd_<16-char-hash>'
    """
    hash_object = hashlib.sha256(text.encode())
    hex_dig = hash_object.hexdigest()
    return 'd_' + hex_dig[:16]


def extract_base_filename(file_path: str) -> str:
    """
    Extract base filename from a file path.
    """
    base_name = os.path.basename(file_path)
    base_name_no_ext = os.path.splitext(base_name)[0]
    return base_name_no_ext


def normalize_file_name(file_name) -> List[str]:
    """
    Normalize file_name to list format.
    
    Provides backward compatibility for old data where file_name was a string
    (single-doc mode). New data always uses list format.
    
    Args:
        file_name: Either a string (old format) or list of strings (new format)
        
    Returns:
        List of file name strings
    """
    if isinstance(file_name, str):
        return [file_name]
    elif isinstance(file_name, list):
        return file_name
    else:
        return [str(file_name)]


def get_file_identifier(file_name_list: List[str]) -> str:
    """
    Get a canonical string identifier from a file_name list.
    
    For single-doc (1-element list), returns the base filename.
    For multi-doc (2+ elements), returns a hash of the sorted paths.
    
    Args:
        file_name_list: List of file names in the bundle
        
    Returns:
        String identifier for use in chunk mappings
    """
    if not file_name_list:
        return ""
    if len(file_name_list) == 1:
        return extract_base_filename(file_name_list[0])
    # Multi-doc: use hash of sorted paths for consistent identifier
    return hashlib.md5("||".join(sorted(file_name_list)).encode()).hexdigest()[:16]


def load_generated_json_files(input_path: str) -> pd.DataFrame:
    """
    Load generated JSON data from either a single file or a folder of batch files.
    
    Args:
        input_path: Path to either:
            - A single merged JSON file
            - A folder containing generated_batch*.json files
        
    Returns:
        Combined DataFrame with all records
    """
    all_records = []
    
    # Check if input is a file or directory
    if os.path.isfile(input_path):
        # Single file mode (merged JSON)
        print(f"Loading single JSON file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
            if isinstance(records, list):
                all_records.extend(records)
            else:
                all_records.append(records)
    else:
        # Folder mode (batch files)
        json_files = sorted(glob.glob(os.path.join(input_path, 'generated_batch*.json')))
        
        if not json_files:
            json_files = sorted(glob.glob(os.path.join(input_path, '*.json')))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {input_path}")
        
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            print(f"  Loading: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
                if isinstance(records, list):
                    all_records.extend(records)
                else:
                    all_records.append(records)
    
    # Normalize file_name to list format (backward compat for old string format)
    print("Normalizing file_name fields...")
    for record in all_records:
        if 'file_name' in record:
            record['file_name'] = normalize_file_name(record['file_name'])
    
    # Filter mismatched records
    print("Filtering mismatched records...")
    all_records, dropped_count = filter_mismatched_records(all_records)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} records with mismatched qa_evaluations/deduplicated_qa_pairs sizes")
    
    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} total records")
    return df


def build_corpus_and_mappings(
    generated_df: pd.DataFrame
) -> Tuple[Dict[str, str], Dict[Tuple[str, int], str]]:
    """
    Build deduplicated corpus and chunk mappings from generated data.
    
    Args:
        generated_df: DataFrame with 'file_name' and 'chunks' columns
        
    Returns:
        Tuple of:
        - corpus: Dict mapping text -> corpus_id (deduplicated by text content)
        - chunk_mapping: Dict mapping (base_filename, chunk_id) -> text
    """
    corpus = {}  # text -> corpus_id
    chunk_mapping = {}  # (base_filename, chunk_id) -> text
    
    print("Building corpus and chunk mappings...")
    
    for _, row in generated_df.iterrows():
        file_name_list = row.get('file_name', [])
        chunks = row.get('chunks', [])
        
        if not chunks or not file_name_list:
            continue
        
        # Get canonical identifier for this file/bundle
        file_identifier = get_file_identifier(file_name_list)
        
        # Handle numpy arrays
        if hasattr(chunks, 'tolist'):
            chunks = chunks.tolist()
        
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk_id = chunk.get('chunk_id')
                text = chunk.get('text', '')
            else:
                chunk_id = getattr(chunk, 'chunk_id', None)
                text = getattr(chunk, 'text', '')
            
            if chunk_id is None or not text:
                continue
            
            # Store mapping using canonical file identifier
            chunk_mapping[(file_identifier, chunk_id)] = text
            
            # Add to deduplicated corpus
            if text not in corpus:
                corpus[text] = get_corpus_id(text)
    
    print(f"Built corpus with {len(corpus)} unique documents from {len(chunk_mapping)} total chunks")
    return corpus, chunk_mapping


def create_train_val_test_split(
    filtered_qa_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split filtered QA pairs into train, val, and test sets by file/bundle.
    
    Args:
        filtered_qa_df: DataFrame with filtered QA pairs
        train_ratio: Ratio of files for training (e.g., 0.8 for 80%)
        val_ratio: Ratio of files for validation (e.g., 0.1 for 10%)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    random.seed(seed)
    
    # Validate ratios
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be <= 1.0")
    
    # Get unique files/bundles - convert lists to tuples for hashability
    # file_name is now always a list (e.g., ['doc.txt'] or ['a.txt', 'b.txt'])
    unique_file_tuples = list(set(
        tuple(f) if isinstance(f, list) else (f,) 
        for f in filtered_qa_df['file_name']
    ))
    
    # Shuffle files
    random.shuffle(unique_file_tuples)
    
    n_train = int(len(unique_file_tuples) * train_ratio)
    n_val = int(len(unique_file_tuples) * val_ratio)
    
    train_files = set(unique_file_tuples[:n_train])
    val_files = set(unique_file_tuples[n_train:n_train + n_val])
    test_files = set(unique_file_tuples[n_train + n_val:])
    
    # Helper to check if a file_name belongs to a set of tuples
    def file_in_set(file_name, file_set):
        file_tuple = tuple(file_name) if isinstance(file_name, list) else (file_name,)
        return file_tuple in file_set
    
    train_df = filtered_qa_df[filtered_qa_df['file_name'].apply(lambda f: file_in_set(f, train_files))]
    val_df = filtered_qa_df[filtered_qa_df['file_name'].apply(lambda f: file_in_set(f, val_files))]
    test_df = filtered_qa_df[filtered_qa_df['file_name'].apply(lambda f: file_in_set(f, test_files))]
    
    print(f"Split: {len(train_files)} train files/bundles ({len(train_df)} QA pairs), "
          f"{len(val_files)} val files/bundles ({len(val_df)} QA pairs), "
          f"{len(test_files)} test files/bundles ({len(test_df)} QA pairs)")
    
    return train_df, val_df, test_df


def generate_training_set(
    corpus: Dict[str, str],
    chunk_mapping: Dict[Tuple[str, int], str],
    train_df: pd.DataFrame,
    output_dir: str,
    corpus_id: str,
    max_pos_docs: int = 5,
    output_filename: str = 'train.json',
    set_name: str = 'training',
    write_corpus: bool = True
) -> None:
    """
    Generate training/validation set in retriever training format.
    
    Args:
        corpus: Dict mapping text -> corpus_id
        chunk_mapping: Dict mapping (base_filename, chunk_id) -> text
        train_df: DataFrame with QA pairs
        output_dir: Output directory path
        corpus_id: Corpus identifier
        max_pos_docs: Maximum number of positive docs per query
        output_filename: Name of the output JSON file (default: 'train.json')
        set_name: Name of the set for logging (default: 'training')
        write_corpus: Whether to write corpus parquet and metadata files (default: True)
    """
    print(f"Generating {set_name} set...")
    
    # Create output directories
    corpus_dir = os.path.join(output_dir, 'corpus')
    os.makedirs(corpus_dir, exist_ok=True)
    
    training_data = []
    question_counter = 0
    skipped_queries = 0
    skipped_too_many_pos = 0
    
    for _, qa_pair in train_df.iterrows():
        file_name_list = qa_pair.get('file_name', [])
        # file_name is now always a list; get canonical identifier
        file_identifier = get_file_identifier(file_name_list) if file_name_list else ''
        segment_ids = qa_pair.get('segment_ids', [])
        question = qa_pair.get('question', '')
        
        if not question:
            skipped_queries += 1
            continue
        
        # Handle numpy arrays
        if hasattr(segment_ids, 'tolist'):
            segment_ids = segment_ids.tolist()
        
        # Skip queries with too many positive docs
        if len(segment_ids) > max_pos_docs:
            skipped_too_many_pos += 1
            continue
        
        # Get corpus IDs for all segments
        pos_docs = []
        all_segments_exist = True
        
        for segment_id in segment_ids:
            key = (file_identifier, segment_id)
            if key not in chunk_mapping:
                all_segments_exist = False
                break
            
            text = chunk_mapping[key]
            corpus_doc_id = corpus[text]
            pos_docs.append({'id': corpus_doc_id})
        
        if not all_segments_exist or not pos_docs:
            skipped_queries += 1
            continue
        
        training_data.append({
            'question_id': f'q{question_counter}',
            'question': question,
            'corpus_id': corpus_id,
            'pos_doc': pos_docs,
            'neg_doc': []
        })
        question_counter += 1
    
    print(f"  Generated {len(training_data)} {set_name} queries")
    if skipped_queries > 0:
        print(f"  Skipped {skipped_queries} queries (missing segments or empty question)")
    if skipped_too_many_pos > 0:
        print(f"  Skipped {skipped_too_many_pos} queries (exceeded max_pos_docs={max_pos_docs})")
    
    # Write output JSON file
    # Use relative path for corpus so data is portable across machines/containers
    train_json_path = os.path.join(output_dir, output_filename)
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'corpus': {'path': './corpus/'},
            'data': training_data
        }, f, indent=2, sort_keys=False)
    
    print(f"  Wrote {train_json_path}")
    
    # Create corpus parquet and metadata (only once, typically for training set)
    if write_corpus:
        corpus_list = [{'id': doc_id, 'text': text} for text, doc_id in corpus.items()]
        corpus_df = pd.DataFrame(corpus_list)
        parquet_path = os.path.join(corpus_dir, 'train.parquet')
        corpus_df.to_parquet(parquet_path, index=False)
        print(f"  Wrote {parquet_path} with {len(corpus_list)} documents")
        
        # Create metadata
        metadata_path = os.path.join(corpus_dir, 'merlin_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'corpus_id': corpus_id,
                'class': 'TextQADataset'
            }, f, indent=2, sort_keys=False)
        
        print(f"  Wrote {metadata_path}")


def generate_eval_set(
    corpus: Dict[str, str],
    chunk_mapping: Dict[Tuple[str, int], str],
    eval_df: pd.DataFrame,
    output_dir: str,
    max_pos_docs: int = 5,
    eval_only: bool = False,
    use_group_id_in_eval: bool = False
) -> None:
    """
    Generate evaluation set in BEIR format.
    
    Args:
        corpus: Dict mapping text -> corpus_id
        chunk_mapping: Dict mapping (base_filename, chunk_id) -> text
        eval_df: DataFrame with QA pairs
        output_dir: Output directory path
        max_pos_docs: Maximum number of positive docs per query
        eval_only: If True, output directly to output_dir instead of eval_beir/ subfolder
        use_group_id_in_eval: If True, use group_id (hash-based) in qrels; otherwise use _id (default)
    """
    print("Generating evaluation set...")
    
    # In eval-only mode, output directly to output_dir; otherwise use eval_beir/ subfolder
    if eval_only:
        eval_dir = output_dir
    else:
        eval_dir = os.path.join(output_dir, 'eval_beir')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Generate corpus.jsonl
    corpus_path = os.path.join(eval_dir, 'corpus.jsonl')
    corpus_id_counter = 0
    text_to_beir_id = {}
    
    with open(corpus_path, 'w', encoding='utf-8') as corpus_file:
        for text, hash_id in corpus.items():
            beir_id = f'd{corpus_id_counter}'
            text_to_beir_id[text] = beir_id
            
            corpus_entry = {
                '_id': beir_id,
                'metadata': {},
                'text': text,
                'title': ''
            }
            
            # Only include group_id if using it in qrels
            if use_group_id_in_eval:
                corpus_entry['group_id'] = hash_id
            
            corpus_file.write(json.dumps(corpus_entry) + '\n')
            corpus_id_counter += 1
    
    print(f"  Wrote {corpus_path} with {corpus_id_counter} documents")
    
    # Generate queries.jsonl and qrels
    queries_path = os.path.join(eval_dir, 'queries.jsonl')
    query_mappings = []
    query_counter = 0
    skipped_queries = 0
    skipped_too_many_pos = 0
    
    with open(queries_path, 'w', encoding='utf-8') as queries_file:
        for _, qa_pair in eval_df.iterrows():
            file_name_list = qa_pair.get('file_name', [])
            # file_name is now always a list; get canonical identifier
            file_identifier = get_file_identifier(file_name_list) if file_name_list else ''
            segment_ids = qa_pair.get('segment_ids', [])
            question = qa_pair.get('question', '')
            
            if not question:
                skipped_queries += 1
                continue
            
            # Handle numpy arrays
            if hasattr(segment_ids, 'tolist'):
                segment_ids = segment_ids.tolist()
            
            # Skip queries with too many positive docs
            if len(segment_ids) > max_pos_docs:
                skipped_too_many_pos += 1
                continue
            
            # Check if all segments exist
            all_segments_exist = True
            for segment_id in segment_ids:
                key = (file_identifier, segment_id)
                if key not in chunk_mapping:
                    all_segments_exist = False
                    break
            
            if not all_segments_exist:
                skipped_queries += 1
                continue
            
            query_id = f'q{query_counter}'
            query_mappings.append((query_id, file_identifier, segment_ids))
            
            # Build metadata
            metadata = {}
            metadata_fields = [
                'query_type', 'reasoning_type', 'hop_count',
                'question_complexity', 'quality_score', 'answer',
                'hop_contexts'
            ]
            
            for field in metadata_fields:
                val = qa_pair.get(field)
                if val is not None:
                    if hasattr(val, 'tolist'):
                        val = val.tolist()
                    metadata[field] = val
            
            metadata['file_name'] = file_name_list
            metadata['segment_ids'] = segment_ids
            
            query_entry = {
                '_id': query_id,
                'metadata': metadata,
                'text': question
            }
            
            queries_file.write(json.dumps(query_entry) + '\n')
            query_counter += 1
    
    print(f"  Wrote {queries_path} with {query_counter} queries")
    if skipped_queries > 0:
        print(f"  Skipped {skipped_queries} queries (missing segments or empty question)")
    if skipped_too_many_pos > 0:
        print(f"  Skipped {skipped_too_many_pos} queries (exceeded max_pos_docs={max_pos_docs})")
    
    # Generate qrels/test.tsv
    qrels_dir = os.path.join(eval_dir, 'qrels')
    os.makedirs(qrels_dir, exist_ok=True)
    
    qrels_path = os.path.join(qrels_dir, 'test.tsv')
    qrels_count = 0
    
    with open(qrels_path, 'w', encoding='utf-8') as qrels_file:
        qrels_file.write('query-id\tcorpus-id\tscore\n')
        
        for query_id, file_identifier, segment_ids in query_mappings:
            for segment_id in segment_ids:
                key = (file_identifier, segment_id)
                text = chunk_mapping[key]
                
                # Use group_id (hash-based) or _id (beir sequential id) based on flag
                if use_group_id_in_eval:
                    doc_id = corpus[text]  # hash-based group_id
                else:
                    doc_id = text_to_beir_id[text]  # sequential _id (e.g., d0, d1, ...)
                
                qrels_file.write(f"{query_id}\t{doc_id}\t1\n")
                qrels_count += 1
    
    id_type = "group_id" if use_group_id_in_eval else "_id"
    print(f"  Wrote {qrels_path} with {qrels_count} mappings (using {id_type})")


def main():
    parser = argparse.ArgumentParser(
        description='Convert SDG output to retriever data formats (training and/or evaluation)'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to merged JSON file or folder containing generated_batch*.json files'
    )
    parser.add_argument(
        '--corpus-id',
        type=str,
        required=True,
        help='Corpus identifier (e.g., "my_corpus")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Path to output directory (default: <input>_train_eval or <input>_eval with --eval-only)'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Generate only evaluation data in BEIR format (no train/val splits, no corpus parquet)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of files for training (default: 0.8, ignored with --eval-only)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Ratio of files for validation (default: 0.1, ignored with --eval-only)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=7.0,
        help='Minimum quality score for QA pairs (default: 7.0)'
    )
    parser.add_argument(
        '--max-pos-docs',
        type=int,
        default=5,
        help='Maximum number of positive docs (segment_ids) per query (default: 5)'
    )
    parser.add_argument(
        '--use-group-id-in-eval',
        action='store_true',
        help='Use group_id (hash-based) instead of _id in qrels (default: use _id)'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    input_path = os.path.abspath(args.input_path)
    if not os.path.exists(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")
    
    # Set default output directory based on input path and mode
    if args.output_dir is None:
        suffix = '_eval' if args.eval_only else '_train_eval'
        if os.path.isfile(input_path):
            # For single file, use filename without extension
            input_basename = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = os.path.join(
                os.path.dirname(input_path),
                f"{input_basename}{suffix}"
            )
        else:
            output_dir = os.path.abspath(input_path.rstrip('/') + suffix)
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SDG to Retriever Data Converter")
    print("=" * 80)
    print(f"Input path: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Corpus ID: {args.corpus_id}")
    if args.eval_only:
        print(f"Mode: Evaluation only (BEIR format)")
    else:
        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        print(f"Mode: Train/Val/Test split")
        print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={test_ratio:.2f}")
    print(f"Random seed: {args.seed}")
    print(f"Quality threshold: {args.quality_threshold}")
    print(f"Max positive docs: {args.max_pos_docs}")
    print(f"Eval qrels ID type: {'group_id' if args.use_group_id_in_eval else '_id'}")
    print()
    
    # Step 1: Load generated data
    print("=" * 80)
    print("Step 1: Loading generated data")
    print("=" * 80)
    generated_df = load_generated_json_files(input_path)
    print()
    
    # Step 2: Build corpus and mappings
    print("=" * 80)
    print("Step 2: Building corpus and mappings")
    print("=" * 80)
    corpus, chunk_mapping = build_corpus_and_mappings(generated_df)
    print()
    
    # Step 3: Filter QA pairs by quality
    print("=" * 80)
    print("Step 3: Filtering QA pairs by quality")
    print("=" * 80)
    filtered_qa_df, skipped_files = filter_qa_pairs_by_quality(generated_df, args.quality_threshold)
    print()
    
    if args.eval_only:
        # Eval-only mode: generate BEIR format directly
        print("=" * 80)
        print("Step 4: Generating evaluation set (BEIR format)")
        print("=" * 80)
        generate_eval_set(corpus, chunk_mapping, filtered_qa_df, output_dir, args.max_pos_docs, eval_only=True, use_group_id_in_eval=args.use_group_id_in_eval)
        print()
        
        print("=" * 80)
        print("Conversion complete!")
        print("=" * 80)
        print(f"Output location: {output_dir}")
        print("Generated (BEIR format):")
        print("  - corpus.jsonl")
        print("  - queries.jsonl")
        print("  - qrels/test.tsv")
    else:
        # Full train/val/test mode
        # Step 4: Create train/val/test split
        print("=" * 80)
        print("Step 4: Creating train/val/test split")
        print("=" * 80)
        train_df, val_df, test_df = create_train_val_test_split(
            filtered_qa_df, args.train_ratio, args.val_ratio, args.seed
        )
        print()
        
        # Step 5: Generate training set
        print("=" * 80)
        print("Step 5: Generating training set")
        print("=" * 80)
        generate_training_set(
            corpus, chunk_mapping, train_df, output_dir, args.corpus_id, args.max_pos_docs,
            output_filename='train.json', set_name='training'
        )
        print()
        
        # Step 6: Generate validation set
        print("=" * 80)
        print("Step 6: Generating validation set")
        print("=" * 80)
        generate_training_set(
            corpus, chunk_mapping, val_df, output_dir, args.corpus_id, args.max_pos_docs,
            output_filename='val.json', set_name='validation', write_corpus=False
        )
        print()
        
        # Step 7: Generate test/evaluation set
        print("=" * 80)
        print("Step 7: Generating test/evaluation set")
        print("=" * 80)
        generate_eval_set(corpus, chunk_mapping, test_df, output_dir, args.max_pos_docs, eval_only=False, use_group_id_in_eval=args.use_group_id_in_eval)
        print()
        
        print("=" * 80)
        print("Conversion complete!")
        print("=" * 80)
        print(f"Output location: {output_dir}")
        print("Generated:")
        print("  - train.json (retriever training format)")
        print("  - val.json (retriever validation format)")
        print("  - corpus/ (parquet + metadata)")
        print("  - eval_beir/ (BEIR test/evaluation format)")
    
    # Print skipped files summary
    if skipped_files:
        print()
        print("=" * 80)
        print(f"Skipped Files ({len(skipped_files)} total)")
        print("=" * 80)
        for item in skipped_files:
            print(f"  - {item['file_name']}: {item['reason']}")


if __name__ == '__main__':
    main()

