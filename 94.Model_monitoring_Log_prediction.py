import pandas as pd
import datetime

def log_prediction(input_data, prediction, source: str):
    log_entry = {
        'timestamp': datetime.datetime.now(),
        'source': source,  # e.g., 'iris' or 'titanic'
        'input': str(input_data),
        'prediction': prediction
    }
    df = pd.DataFrame([log_entry])
    df.to_csv('predictions_log.csv', mode='a', header=not pd.io.common.file_exists('predictions_log.csv'), index=False)

# Example usage for Iris:
log_prediction([5.1, 3.5, 1.4, 0.2], "setosa", source='iris')

# Example usage for Titanic:
log_prediction({'Pclass': 3, 'Sex': 'male', 'Age': 22, 'Fare': 7.25}, "not survived", source='titanic')
