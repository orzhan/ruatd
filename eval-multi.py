from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging

model_args = ClassificationArgs()
model_args.eval_batch_size=32
model_args.use_multiprocessing = False
model_args.output_dir = '/content/drive/MyDrive/ruarg'
model_args.save_optimizer_and_scheduler = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False


model = ClassificationModel(
    "roberta", "orzhan/ruroberta-ruatd-multi", num_labels=14, args=model_args, 
)

df_test = pd.read_csv('multi/test.csv')

predictions, raw_outputs = model.predict(df_test['Text'].values.tolist())

df_submit = df_test[['Text']].copy()
df_submit['Class'] = [labels[x] for x in predictions]

df_submit['Id'] = df_test['Id']

df_submit[['Id','Class']].to_csv('submit.csv', index=False)

