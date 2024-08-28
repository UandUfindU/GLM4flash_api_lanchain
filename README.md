# GLM4flash_api_langchain

## 这是一个懒人包，可以供你迅速调用最新最潮的GLM4 Flash（免费）

### 如何快速使用？
1. 安装依赖（命令行中）
```bash
pip install -r requirements.txt
```

2. 在ZhipuFreeOhYeah.py中填入你的API（在这里注册后获取api_key：https://open.bigmodel.cn/usercenter/apikeys）
```python
glm_flash_llm = GLMFlash_LLM(api_key="your_api_key")
```

3. 启动脚本，本地命令行调用
```bash
python ZhipuFreeOhYeah.py
```

### 架构说明
标准langchain架构，方便各式工程应用的搭建（agent、RAG等）

实现调用逻辑位于LLM.py中，同时提供deepseek的启动入口

参数调整，模型更换（将涉及付费）可以在LLM.py中进行调整
