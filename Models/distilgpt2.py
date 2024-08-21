from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

print("Chat with the model (type 'exit' to stop):")
while True:
    text = input("You: ")
    if text.lower() == 'exit':
        break

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to("cpu")
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
        top_k=500
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(f"Model: {response}")

