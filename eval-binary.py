from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.nn import Softmax

model_args = ClassificationArgs()
model_args.eval_batch_size=16
model_args.use_multiprocessing = False
model_args.save_optimizer_and_scheduler = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False


model = ClassificationModel(
    "roberta", "orzhan/ruroberta-ruatd-binary", num_labels=2, args=model_args, 
)


df_test = pd.read_csv('binary/test.csv')
predictions, raw_outputs = model.predict(df_test['Text'].values.tolist())
probas = Softmax(dim=-1)(torch.tensor(raw_outputs))
probas1=probas[:,1].numpy()
preds = [1 if x > 0.7 else 0 for x in probas1]

df_submit = df_test[['Text']].copy()
df_submit['Class'] = ['H' if x == 1 else 'M' for x in preds]
df_submit['Id'] = df_test['Id']

df_submit[['Id','Class']].to_csv('submit.csv', index=False)
