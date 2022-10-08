from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from log_handler.Setup import logger

def load_tokenizer_model():
    autotokenizer=AutoTokenizer.from_pretrained("./models", use_fast=True, return_tensors="pt")
    logger.info("Tokenier loaded!")
    automodel=AutoModelForSequenceClassification.from_pretrained("./models", ignore_mismatched_sizes=True)
    logger.info("Model loaded!")
    return autotokenizer, automodel

def load_pipeline(model_path="./models", tokenizer_path="./models"):
    stance_task=pipeline('text-classification',
                     model_path,
                     tokenizer_path,
                     return_all_scores=False)
    logger.info("Pipeline is loaded!")
    return stance_task