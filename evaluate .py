from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

#  Load FIRST 1000 examples & split
dataset = load_dataset("MuskumPillerum/General-Knowledge")
dataset = dataset["train"].select(range(1000)).train_test_split(test_size=0.1, shuffle=False)

test_data = dataset["test"]

# pick first 3 test examples
questions = ["question: " + str(test_data[i]["Question"]) for i in range(3)]
references = [test_data[i]["Answer"] for i in range(3)]

#  Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def generate_answer(model_path, question):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=256).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4
        )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred

print("\n=== BASE MODEL RESPONSES ===\n")
for i, q in enumerate(questions):
    pred = generate_answer("google/flan-t5-small", q)
    print(f"Q{i+1}: {q}")
    print(f"Base Answer       : {pred}")
    print(f"Reference Answer  : {references[i]}")
    print("---")

print("\n=== FINE-TUNED MODEL RESPONSES ===\n")
for i, q in enumerate(questions):
    pred = generate_answer("./flan-t5-small-finetuned", q)
    print(f"Q{i+1}: {q}")
    print(f"Fine-Tuned Answer : {pred}")
    print(f"Reference Answer  : {references[i]}")
    print("---")