from langchain.llms.base import LLM
from typing import Any, List, Optional
from openai import OpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from datetime import datetime  # 引入datetime模块

class DeepSeek_LLM(LLM):
    client: OpenAI = None

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print("DeepSeek API 客户端已初始化")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        try:
            # 记录开始时间
            start_time = datetime.now()

            messages = [
                {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            
            # 记录结束时间
            end_time = datetime.now()
            
            # 计算总响应时间
            total_time = end_time - start_time
            
            # 打印总响应时间
            print(f"Model response time: {total_time}")

            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error: {e}")
            return "Error in DeepSeek API call."

    @property
    def _llm_type(self) -> str:
        return "DeepSeek_LLM"

from langchain.llms.base import LLM
from typing import Any, List, Optional
from zhipuai import ZhipuAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from datetime import datetime

class GLMFlash_LLM(LLM):
    client: ZhipuAI = None

    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn"):
        super().__init__()
        self.client = ZhipuAI(api_key=api_key)
        print("ZhipuAI 客户端已初始化")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        try:
            # 记录开始时间
            start_time = datetime.now()

            messages = [
                {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model="glm-4-flash",  # 使用GLM-4-flash模型
                messages=messages,
                stream=False,
                max_tokens=1024,  # 你可以根据需要调整
                temperature=0.95,  # 可选参数
                top_p=0.7,  # 可选参数
                stop=stop  # 停止词
            )
            
            # 记录结束时间
            end_time = datetime.now()
            
            # 计算总响应时间
            total_time = end_time - start_time
            
            # 打印总响应时间
            print(f"Model response time: {total_time}")

            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error: {e}")
            return "Error in GLM-4-flash API call."

    @property
    def _llm_type(self) -> str:
        return "GLMFlash_LLM"




