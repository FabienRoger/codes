
import asyncio
import base64
import random
from typing import Literal
import attrs
import numpy as np
import openai
from openai.embeddings_utils import cosine_similarity
from dotenv import load_dotenv

from caching import add_to_cache, get_cache, shorten_key

load_dotenv()
Role = Literal["user", "system", "assistant"]

@attrs.define
class Sleeper:
    retry_time: float = 0.5
    min_retry_time: float = 0.1
    max_retry_time: float = 120
    increase_retry_factor: float = 1.2
    decrease_retry_factor: float = 0.95
    min_rdm_frac: float = 0.2
    
    def decrease_sleep(self):
        self.retry_time = max(self.retry_time * self.decrease_retry_factor, self.min_retry_time)
    
    def increase_sleep(self):
        self.retry_time = min(self.retry_time * self.increase_retry_factor, self.max_retry_time)
    
    async def sleep(self):
        await asyncio.sleep(self.retry_time * random.uniform(self.min_rdm_frac, 1))

@attrs.define
class OpenAIChatModel:
    model_id: str
    sleeper: Sleeper = attrs.field(factory=Sleeper)
    
    async def __call__(self, messages: list[tuple[Role, str]], request_timeout: float=10, **kwargs) -> list[str]:
        k = shorten_key(str(messages) + str(sorted(list(kwargs.items()))))
        cache_name = f"oai-{self.model_id}"
        
        while True:
            try:
                if k in get_cache(cache_name):
                    return get_cache(cache_name)[k]
                
                api_response = await openai.ChatCompletion.acreate(
                    messages=[
                        {"role": role, "content": content}
                        for role, content in messages
                    ],
                    model=self.model_id,
                    **kwargs,
                    request_timeout=request_timeout,
                )
                completions = [message["message"]["content"] for message in api_response["choices"]]
                self.sleeper.decrease_sleep()
                break
            except Exception as e:
                print(f"OpenAI API error: {e}")
                self.sleeper.increase_sleep()
                await self.sleeper.sleep()
        
        add_to_cache(cache_name, k, completions)
        return completions


def float_list_to_compact_str(float_list):
    arr = np.array(float_list, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode('ascii')

def compact_str_to_float_list(compact_str):
    arr_bytes = base64.b64decode(compact_str)
    return np.frombuffer(arr_bytes, dtype=np.float32).tolist()

@attrs.define
class OpenAIEmbeddingModel:
    model_id: str
    sleeper: Sleeper = attrs.field(factory=Sleeper)
    
    async def __call__(self, s: str, **kwargs) -> list[float]:
        if len(s) == 0:
            return
        
        k = shorten_key(s + str(sorted(list(kwargs.items()))))
        cache_name = f"oaie-{self.model_id}"
        
        while True:
            try:
                if k in get_cache(cache_name):
                    return compact_str_to_float_list(get_cache(cache_name)[k])
                
                api_response = await openai.Embedding.acreate(
                    input=[s], engine=self.model_id,
                    **kwargs,
                )
                result = api_response["data"][0]["embedding"]
                self.sleeper.decrease_sleep()
                break
            except Exception as e:
                print(f"OpenAI API error: {e}")
                self.sleeper.increase_sleep()
                await self.sleeper.sleep()
        
        add_to_cache(cache_name, k, float_list_to_compact_str(result))
        return result

    @staticmethod
    def cosine_similarity(e1: list[float], e2: list[float]) -> float:
        return cosine_similarity(e1, e2)
    
gpt_4o = OpenAIChatModel("gpt-4o")
ada_embedding = OpenAIEmbeddingModel("text-embedding-3-large")

async def test():
    print(await gpt_4o([("user", "Hello!")]))
    r1 = await ada_embedding("Hello!")
    print(await gpt_4o([("user", "Hesssllo!")]))
    r2 = await ada_embedding("Hi!")
    r3 = await ada_embedding("Hello!")
    print(ada_embedding.cosine_similarity(r1, r2))
    print(ada_embedding.cosine_similarity(r1, r3))
    print(ada_embedding.cosine_similarity(r2, r3))



if __name__ == "__main__":
    asyncio.run(test())