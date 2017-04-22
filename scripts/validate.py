import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

TRUE_DATA_PATH = '../data/test_sample.csv'
COLUMN_NAME = 'is_listened'

def validate(filename):
  true_data = pd.read_csv(TRUE_DATA_PATH)
  result_data = pd.read_csv(filename)
  return roc_auc_score(true_data[COLUMN_NAME], result_data[COLUMN_NAME])

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print('usage: python validate.py [result.csv]')
    sys.exit()
  print(validate(sys.argv[1]))