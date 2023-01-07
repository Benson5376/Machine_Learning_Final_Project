import csv
import pandas as pd
from scipy.stats import rankdata
submission = pd.read_csv('best_model.csv')

submission.head()

submission['rank0'] = rankdata(submission['lr0'])
submission['rank1'] = rankdata(submission['lr1'])
submission['rank2'] = rankdata(submission['lr2'])
submission['rank3'] = rankdata(submission['lr3'])

submission['failure'] = submission['rank0']*0.25 + submission['rank1']*0.25 + submission['rank2']*0.25 + submission['rank3']*0.25

submission.head()

submission[['id', 'failure']].to_csv('submission.csv', index=False)