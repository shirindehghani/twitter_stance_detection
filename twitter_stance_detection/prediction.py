from transformers import pipeline
from log_handler.Setup import logger

import warnings

warnings.filterwarnings('ignore')

def prediction(data, stance_task):
    data['prediction']=data['preprocessed_text'].apply(lambda x: stance_task(x)[0]['label'])
    logger.info("Prediction finished!")
    data['prediction']=data['prediction'].apply(lambda x: 0 if x=="LABEL_0" else 1 if x=="LABEL_1" else 2)
    return data['prediction'].values.tolist()