from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

#  Load FIRST 1000 examples & split
dataset = load_dataset("MuskumPillerum/General-Knowledge")
dataset = dataset["train"].select(range(1000)).train_test_split(test_size=0.1, shuffle=False)

#  Load tokenizer & model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#  Preprocessing
def preprocess(batch):
    inputs = ["question: " + str(q) for q in batch["Question"]]
    answers = [str(a) for a in batch["Answer"]]

    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(answers, max_length=64, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False
)

#  Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./flan-t5-small-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    predict_with_generate=True,
    fp16=False,
    save_total_limit=2
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#  Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./flan-t5-small-finetuned")
tokenizer.save_pretrained("./flan-t5-small-finetuned")