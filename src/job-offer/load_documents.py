import boto3
import json
import random
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class JobContext(BaseModel):
    job_title: str
    language: str
    skills: Optional[List[str]] = []


def load_job_contexts_from_s3(bucket_name: str, file_key: str) -> List[JobContext]:
    """Load job contexts dataset from S3"""
    # Boto3 will automatically use credentials from AWS CLI, environment, or IAM role
    s3 = boto3.client('s3')
    
    # Download the dataset from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = json.loads(response['Body'].read().decode('utf-8'))
    
    # Convert to JobContext objects
    job_contexts = []
    for item in data:
        context = JobContext(
            job_title=item['context']['job_title'],
            language=item['context']['language'],
            skills=item['context'].get('skills', [])
        )
        job_contexts.append(context)
    
    return job_contexts


def load_documents() -> Tuple[List[JobContext], List[JobContext]]:
    """Load job contexts for training and validation from S3"""
    
    # S3 configuration
    bucket_name = "job-offer-generation"
    file_key = "datasets/job_offer_dataset.json"
    
    # Load all job contexts from S3
    all_contexts = load_job_contexts_from_s3(bucket_name, file_key)
    
    # Shuffle with fixed seed for reproducibility
    random.seed(80)
    random.shuffle(all_contexts)
    
    # Split into validation and training sets
    # Use 10% for validation, 90% for training by default
    total_samples = len(all_contexts)
    val_size = int(os.getenv("VAL_SIZE", str(total_samples // 10)))  # 10% for validation
    train_size = int(os.getenv("TRAIN_SIZE", str(total_samples - val_size)))  # Rest for training
    
    if train_size + val_size > len(all_contexts):
        raise ValueError(
            f"Train size + val size ({train_size + val_size}) is greater than "
            f"the total number of job contexts ({len(all_contexts)})"
        )
    
    val_contexts = all_contexts[:val_size]
    train_contexts = all_contexts[val_size : val_size + train_size]
    
    print(f"Loaded {len(all_contexts)} job contexts from S3")
    print(f"Train set size: {len(train_contexts)}")
    print(f"Val set size: {len(val_contexts)}")
    
    return val_contexts, train_contexts
