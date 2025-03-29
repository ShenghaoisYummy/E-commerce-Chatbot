# Function to get model responses for validation data
def get_model_responses(fine_tuned_model, validation_data):
    responses = []
    for example in tqdm(validation_data, desc="Getting model responses"):
        user_message = example['messages'][1]['content']  # User message
        try:
            response = openai.chat.completions.create(
                model=fine_tuned_model,
                messages=[
                    {"role": "system", "content": "You are a helpful e-commerce assistant."},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,  # Use 0 for most deterministic response
                max_tokens=150
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error getting response: {e}")
            responses.append("")
    return responses

# Get the fine-tuned model name from the job status
fine_tuned_model = job_status.fine_tuned_model

# Get model responses for validation data
model_responses = get_model_responses(fine_tuned_model, val_data)

# Prepare references and hypotheses for BLEU calculation
references = []
for example in val_data:
    references.append([example['messages'][2]['content'].split()])  # Split into words

hypotheses = [response.split() for response in model_responses]

# Calculate BLEU using NLTK
smoother = SmoothingFunction().method1
bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoother)
print(f"BLEU Score (NLTK): {bleu_score:.4f}")

# Calculate BLEU using sacrebleu (more standard in MT evaluation)
sacrebleu = BLEU()
references_sacrebleu = [[example['messages'][2]['content']] for example in val_data]
hypotheses_sacrebleu = model_responses

# Convert to the format expected by sacrebleu
references_transposed = [list(refs) for refs in zip(*references_sacrebleu)]
bleu_scores_sacrebleu = []

for i in range(len(hypotheses_sacrebleu)):
    result = sacrebleu.corpus_score([hypotheses_sacrebleu[i]], [[references_sacrebleu[i][0]]])
    bleu_scores_sacrebleu.append(result.score)

avg_sacrebleu = sum(bleu_scores_sacrebleu) / len(bleu_scores_sacrebleu)
print(f"Average SacreBLEU Score: {avg_sacrebleu:.4f}")