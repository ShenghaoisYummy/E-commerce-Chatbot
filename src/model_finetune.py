import openai
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sacrebleu import BLEU

def create_fine_tuning_job(
    training_file_id: str,
    validation_file_id: str,
    model: str = "gpt-3.5-turbo",
    n_epochs: int = 1
) -> str:
    """
    Creates an OpenAI fine-tuning job with specified parameters.
    
    Args:
        training_file_id (str): ID of the training file uploaded to OpenAI
        validation_file_id (str): ID of the validation file uploaded to OpenAI
        model (str): Base model to fine-tune (default: "gpt-3.5-turbo")
        n_epochs (int): Number of training epochs (default: 1)
    
    Returns:
        str: ID of the created fine-tuning job
    
    Raises:
        Exception: If the fine-tuning job creation fails
    """
    try:
        fine_tuning_job = openai.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model,
            hyperparameters={
                "n_epochs": n_epochs
            }
        )
        print(f"Fine-tuning job created: {fine_tuning_job.id}")
        return fine_tuning_job
        
    except Exception as e:
        print(f"Error creating fine-tuning job: {str(e)}")
        raise

# Check the status of your fine-tuning job
def check_fine_tuning_status(job_id):
    job = openai.fine_tuning.jobs.retrieve(job_id)
    print(f"Status: {job.status}")
    print(f"Trained tokens: {job.trained_tokens}")
    if job.status == "succeeded":
        print(f"Fine-tuned model: {job.fine_tuned_model}")
    return job

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


def evaluate_model_bleu(fine_tuned_model, evl_data, get_model_responses_func):
    """
    Evaluates a fine-tuned model using BLEU metrics (both NLTK and SacreBLEU implementations).
    
    Args:
        job_status: Object containing fine-tuned model information
        val_data: List of validation examples with message content
        get_model_responses_func: Function that generates model responses given model and data
    
    Returns:
        tuple: (NLTK BLEU score, Average SacreBLEU score)
    
    Raises:
        Exception: If evaluation fails due to invalid data or model issues
    """
    try:

        if not fine_tuned_model:
            raise ValueError("No fine-tuned model found in job status")

        # Get model responses
        model_responses = get_model_responses_func(fine_tuned_model, evl_data)
        
        # Prepare data for NLTK BLEU
        references = []
        for example in evl_data:
            if len(example['messages']) < 3:
                raise ValueError("Validation data missing expected message content")
            references.append([example['messages'][2]['content'].split()])
        
        hypotheses = [response.split() for response in model_responses]
        
        # Calculate NLTK BLEU
        smoother = SmoothingFunction().method1
        bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoother)
        print(f"BLEU Score (NLTK): {bleu_score:.4f}")
        
        # Prepare data for SacreBLEU
        references_sacrebleu = [[example['messages'][2]['content']] for example in val_data]
        hypotheses_sacrebleu = model_responses
        
        # Calculate SacreBLEU scores
        sacrebleu = BLEU()
        bleu_scores_sacrebleu = []
        
        for i in range(len(hypotheses_sacrebleu)):
            result = sacrebleu.corpus_score(
                [hypotheses_sacrebleu[i]], 
                [[references_sacrebleu[i][0]]]
            )
            bleu_scores_sacrebleu.append(result.score)
        
        avg_sacrebleu = sum(bleu_scores_sacrebleu) / len(bleu_scores_sacrebleu)
        print(f"Average SacreBLEU Score: {avg_sacrebleu:.4f}")
        
        return bleu_score, avg_sacrebleu
        
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        raise

# Example usage:
# Assuming you have:
# - job_status from fine-tuning job
# - val_data as list of examples with messages
# - get_model_responses function defined
#
# def get_model_responses(model_name, data):
#     # Your implementation here
#     pass
#
# bleu_nltk, bleu_sacre = evaluate_model_bleu(job_status, val_data, get_model_responses)