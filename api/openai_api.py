import openai

def call(prompt, model, api_key):
    return openai.ChatCompletion.create(
        model=model,
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    ).choices[0].message.content.strip()
