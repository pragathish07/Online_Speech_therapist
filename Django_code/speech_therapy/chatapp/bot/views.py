from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from huggingface_hub import InferenceClient
import random
import re  # Import re for regular expressions
import textwrap  # Import textwrap for formatting

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
API_KEY = "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"
headers = {"Authorization": f"Bearer {API_KEY}"}

def format_prompt(message, custom_instructions=None):
    prompt = ""
    if custom_instructions:
        prompt += f" [INST] {custom_instructions} [/INST]"
    prompt += f" [INST] {message} [/INST]"
    return prompt

def generate(prompt, temperature=0.9, max_new_tokens=512, top_p=0.95, repetition_penalty=1.0):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=random.randint(0, 10**7),
    )
    
    custom_instructions = "consider yourself a speech therapist and respond to the user as if you are a therapist."
    formatted_prompt = format_prompt(prompt, custom_instructions)
    
    client = InferenceClient(API_URL, headers=headers)
    response = client.text_generation(formatted_prompt, **generate_kwargs)
    
    if isinstance(response, list):
        bot_response = response[0]["choices"][0]["text"] if response else ""
    elif isinstance(response, dict):
        bot_response = response["choices"][0]["text"] if "choices" in response else response.get("text", "")
    else:
        bot_response = str(response)
    
    # Use regular expression to find numbers and add line breaks
    bot_response = re.sub(r'\b(\d+)\b', r'\1\n', bot_response)
    
    # Format the bot response using textwrap
    formatted_bot_response = "\n".join(textwrap.wrap(bot_response, width=80))
    
    return formatted_bot_response

@csrf_exempt
def home(request):
    if request.method == "POST":
        user_prompt = request.POST.get("user_prompt", "")
        generated_text = generate(user_prompt)
        return JsonResponse({"bot_response": generated_text})
    return render(request, "index.html")
