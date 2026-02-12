import os
from groq import Groq
from dotenv import load_dotenv
from index import cfg

load_dotenv()

class RagGenerator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = cfg.get('models', {}).get('generator', 'llama3-8b-8192')

    def generate_answer(self, query, context):
        prompt = f"""
        Use the following pieces of retrieved context to answer the user's question.
        If you don't know the answer based on the context, just say that you don't know.
        
        CONTEXT:
        {context}
        
        QUESTION: 
        {query}
        
        ANSWER:
        """
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Keep it low for factual consistency
        )
        return completion.choices[0].message.content