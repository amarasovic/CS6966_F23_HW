import argparse
import numpy as np
import os 
import jsonlines 
import datasets, transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import random 
from tqdm import tqdm
import torch

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    results = metric.compute(references=labels, predictions=predictions)
    return results

def sentiment_classifier(args):
    '''
    When a dataset has only train and test split, it is good to reserve some portion of the train data for development
    NEVER use the test set for making ANY modeling choice
    '''

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint) 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels)
    model_name = args.model_checkpoint.split("/")[-1]

    if args.do_train:
        '''
        Instead of training for epochs (each epoch iterates through the entire data), we might want to 
        train for a certain number of steps (one step is only one batch). This is helpful when we have 
        already well pretrained model for the task. 
        If I don't need all the training data, I could save more for the development, but since the choice
        of HPs is not crucial here, it doesn't really matter.

        E.g. if we use batch_size=2, eval_steps=25, we observe these accuracies on the dev set of 2.5K instances::
            {'eval_loss': 0.6472113132476807, 'eval_accuracy': 0.6144, 'eval_runtime': 62.0133, 'eval_samples_per_second': 80.628, 'eval_steps_per_second': 40.314, 'epoch': 0.01}  
            {'eval_loss': 0.3902832865715027, 'eval_accuracy': 0.8692, 'eval_runtime': 162.1462, 'eval_samples_per_second': 30.836, 'eval_steps_per_second': 15.418, 'epoch': 0.02} 
            {'eval_loss': 0.22414346039295197, 'eval_accuracy': 0.9268, 'eval_runtime': 62.1229, 'eval_samples_per_second': 80.486, 'eval_steps_per_second': 40.243, 'epoch': 0.03}
        That is, after seeing only 2*3*25=150 train instances, we get 92.7% accuracy on the dev set, so we don't need the entire the entire epoch (20K train instances).
        '''

        train_set = load_dataset(args.task_name, split='train')
        train_val_set = train_set.train_test_split(test_size=args.train_frac_for_dev, stratify_by_column="label")

        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint) 

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

        encoded_train_val_set = train_val_set.map(preprocess_function, batched=True)

        max_steps = 0.2 * len(train_val_set["train"]['text'])// (args.batch_size) # I'll use 20% of the train data only
        eval_steps = int(0.25 * max_steps) 
        train_args = TrainingArguments(os.path.join(args.output_dir, f"{model_name}-finetuned-{args.task_name}"),
                                evaluation_strategy = "steps", 
                                save_strategy = "steps", 
                                learning_rate=args.learning_rate,
                                per_device_train_batch_size=args.batch_size,
                                per_device_eval_batch_size=args.batch_size,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                weight_decay=args.weight_decay,
                                load_best_model_at_end=True, # Makes sure the best model checkpoint is always kept.
                                save_total_limit=args.save_total_limit, # If set to one and `load_best_model_at_end=True` then only one BEST checkpoint is saved. Strongly recommend this because checkpoints take space.
                                metric_for_best_model="accuracy",
                                push_to_hub=False,
                                eval_steps=eval_steps,
                                max_steps=max_steps
                                )
        '''
        If you are training with epochs (which is fine and something you likely did because that is what the notebook did):
        args = TrainingArguments(os.path.join(args.output_dir, f"{model_name}-finetuned-{args.task_name}"),
                                evaluation_strategy = "epoch", #Evaluation is done at the end of each epoch.
                                save_strategy = "epoch", #Save is done at the end of each epoch.
                                learning_rate=args.learning_rate,
                                per_device_train_batch_size=args.batch_size,
                                per_device_eval_batch_size=args.batch_size,
                                num_train_epochs=args.epochs,
                                #gradient_accumulation_steps=args.gradient_accumulation_steps,
                                weight_decay=args.weight_decay,
                                load_best_model_at_end=True,
                                metric_for_best_model="accuracy",
                                push_to_hub=False,
                                save_total_limit=args.save_total_limit
                                )
        '''

        print ('Training the model...')
        trainer = Trainer(model,
                          train_args,
                          train_dataset=encoded_train_val_set["train"],
                          eval_dataset=encoded_train_val_set["test"],
                          tokenizer=tokenizer,
                          compute_metrics=compute_metrics
                         )

        trainer.train()

        '''
        You can run here the accuracy evaluation on the test set with `trainer.evaluate`, but I do not recommend it.
        Remember, the test set is only for the FINAL evaluation of the model.
        You most likely won't train only a single model version.  
        '''

    if args.analyze:
        '''
        Notice that I am using the development set for an analysis because I might form hypotheses about how to change the model
        If I had used the test set, and then based on my analysis changed the model, that would be cheating
        '''

        # This will work only if do_train is True

        predictions = trainer.predict(encoded_train_val_set["test"])
        predicted_labels = np.argmax(predictions.predictions, axis=1)
        correct_labels = encoded_train_val_set["test"]["label"]

        # Indices of instances whose predicted label is different from the correct label
        incorrect_predictions_indices = np.argwhere(predicted_labels != correct_labels)
        analysis_indices = np.random.choice(incorrect_predictions_indices.flatten(),args.analysis_sample_size)
        label_int2str = {0: "negative", 1: "positive"}
        analysis_data = [{'review': encoded_train_val_set["test"]["text"][idx],
                        'label': label_int2str[encoded_train_val_set["test"]["label"][idx]],
                        'predicted': label_int2str[predicted_labels[idx]]} for idx in analysis_indices]

        with jsonlines.open(args.analysis_file, mode='w') as writer:
            for item in analysis_data:
                writer.write(item)

    if args.do_test_eval: 
        test_set = load_dataset(args.task_name, split='test').shuffle(seed=42).select(range(10))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #OPTION 1 (least abstract)
        model.to(device)
        predicted_labels = []
        for inst in tqdm(test_set):
            inputs = tokenizer(inst["text"],return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_labels.append(logits.argmax().item())
        results = metric.compute(predictions=predicted_labels, references=test_set["label"])
        print (results) 

        # OPTION 2 (more abstract)
        pipe = transformers.pipeline("text-classification", 
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    device=device
                                    )
        label_dict = {'LABEL_0': 0, 'LABEL_1': 1}

        from transformers.pipelines.pt_utils import KeyDataset
        predicted_labels = [label_dict[out['label']] for out in pipe(KeyDataset(test_set, "text"),batch_size=args.batch_size)]
        results = metric.compute(predictions=predicted_labels, references=test_set["label"])
        print (results)

        # OPTION 3 (most abstract)
        from evaluate import evaluator
        task_evaluator = evaluator("text-classification")
        results = task_evaluator.compute(
                                        model_or_pipeline=pipe,
                                        data=test_set,
                                        label_mapping={'LABEL_0': 0, 'LABEL_1': 1}
                                        )
        print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='', type=str, help='Directory where model checkpoints will be saved')
    parser.add_argument('--task_name', default="imdb",  type=str, help='Directory where model checkpoints will be saved')    
    parser.add_argument('--model_checkpoint', default="microsoft/deberta-v3-base",  type=str, help='The hf name of the pretrained model we are finetuning OR the path to your finetuned model')    
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')    
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='Read what this is and why it is helpful here: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')    
    parser.add_argument('--save_total_limit', default=1, type=int, help='Number of checkpoints to save')
    parser.add_argument('--analysis_sample_size', default=10, type=int, help='Number of instances to analyze')
    parser.add_argument('--train_frac_for_dev', default=0.02, type=float, help='Fraction of train set to use for development')
    parser.add_argument('--analysis_file', default="out/a1_analysis_data.jsonl", type=str, help='Path to a file where analysis data will be saved')    
    parser.add_argument('--do_train', action='store_true', help='Whether to run training')
    parser.add_argument('--analyze', action='store_true', help='Whether to run the analysis')
    parser.add_argument('--do_test_eval', action='store_true', help='Whether to eval the model on the test set; If used make sure that you are setting arg.model_checkpoint to the path of your finetuned model if you want to eval it')
    args = parser.parse_args()

    sentiment_classifier(args)
