from transformers import Trainer,TrainingArguments
from log_handler.Setup import logger

import warnings

warnings.filterwarnings('ignore')

def fit_model(model,tokenizer,train_dataset,val_dataset,num_train_epochs,per_device_train_batch_size,per_device_eval_batch_size,warmup_steps,weight_decay,logging_steps,learning_rate,save_model_path):
    logger.info("model is loaded!")
    training_args=TrainingArguments(output_dir='./results',num_train_epochs=num_train_epochs,learning_rate=learning_rate,per_device_train_batch_size=per_device_train_batch_size,per_device_eval_batch_size=per_device_eval_batch_size, warmup_steps=warmup_steps,weight_decay=weight_decay, logging_dir='./logs', logging_steps=logging_steps)
    logger.info('training arguments are loaded!')
    trainer=Trainer(args=training_args,train_dataset=train_dataset,eval_dataset=val_dataset,model=model,tokenizer=tokenizer)
    trainer.train()
    logger.info("Training finished!")
    trainer.save_model(save_model_path)
    logger.info("model saved!")