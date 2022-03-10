from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


df_train = pd.read_csv('/content/multi/train.csv', encoding='utf-8')
df_val = pd.read_csv('/content/multi/val.csv', encoding='utf-8')


labels = df_train['Class'].unique().tolist()
df_train['labels'] = [labels.index(x) for x in df_train['Class']]
df_train['text'] = df_train['Text']

df_val['labels'] = [labels.index(x) for x in df_val['Class']]
df_val['text'] = df_val['Text']

df_train = df_train[['text','labels']]
df_val = df_val[['text', 'labels']]

print(df_train['labels'].value_counts())
print(df_val['labels'].value_counts())



logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = ClassificationArgs()
model_args.num_train_epochs = 3
model_args.no_save = False
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_during_training_steps = 1000
model_args.train_batch_size=12
model_args.eval_batch_size=12
model_args.overwrite_output_dir = True
model_args.fp16 = False
model_args.wandb_project = 'ruatd'
model_args.use_multiprocessing = False
model_args.learning_rate = 1e-5
model_args.weight_decay = 0.01
model_args.label_smoothing_factor =0.1

model_args.no_cache=True


model_args.output_dir = '/content/drive/MyDrive/ruatd/'
model_args.save_optimizer_and_scheduler = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False


model = ClassificationModel(
    "roberta", "sberbank-ai/ruRoberta-large", num_labels=14, args=model_args, 
)

from sklearn.metrics import accuracy_score, f1_score

def macro_f1_score(true, pred):
  return f1_score(true, pred, average='macro')

def micro_f1_score(true, pred):
  return f1_score(true, pred, average='micro')


model.train_model(df_train, eval_df=df_val, accuracy_score=accuracy_score, macro_f1_score=macro_f1_score, micro_f1_score=micro_f1_score)

result, model_outputs, wrong_predictions = model.eval_model(df_val)


