import asyncio
import openai


class TaskExecutor:
    def __init__(self, model="gpt-4o-mini"):
        self.prompts = []
        self.model = model
        self.api_key = ""

    def add_task(self, prompt):
        self.prompts.append(prompt)

    async def fetch_completion(self, client, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that detects theses discussed in the text."},
            {"role": "user", "content": prompt}
        ]
        
        completion = await client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=messages
        )
        return completion

    async def execute(self):
        batch_size = 7
        results = []
        client = openai.AsyncOpenAI(api_key=self.api_key)

        for i in range(0, len(self.prompts), batch_size):
            batch = self.prompts[i:i + batch_size]
            tasks = [self.fetch_completion(client, prompt) for prompt in batch]
            responses = await asyncio.gather(*tasks)
            for response in responses:
                message_content = response.choices[0].message.content
                results.append(message_content)

        self.prompts = []

        return results
