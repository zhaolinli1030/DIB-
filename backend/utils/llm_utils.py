import httpx
import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI

from ..config import settings

async def get_llm_response(prompt: str, model: str = None) -> str:
    """
    使用DeepSeek API获取LLM响应
    
    Args:
        prompt: 输入提示
        model: 模型名称（可选，默认使用配置中的模型）
        
    Returns:
        LLM的文本响应
    """
    api_key = settings.DEEPSEEK_API_KEY
    base_url = settings.DEEPSEEK_BASE_URL
    model = model or settings.DEEPSEEK_MODEL
    
    if not api_key:
        # 如果API密钥未设置，返回示例响应
        return _get_mock_response(prompt)
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个数据分析助手。请始终以JSON格式返回你的响应，确保响应是有效的JSON字符串。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=4096  # 默认4K tokens
        )
        
        # 获取响应内容
        content = response.choices[0].message.content
        
        # 清理响应内容
        content = content.strip()
        
        # 尝试提取JSON内容
        try:
            # 如果响应包含markdown代码块，提取其中的JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # 验证JSON格式
            json.loads(content)
            return content
        except json.JSONDecodeError as e:
            print(f"LLM返回的内容不是有效的JSON: {content}")
            print(f"错误详情: {e}")
            # 如果JSON解析失败，返回模拟响应
            return _get_mock_response(prompt)
        
    except Exception as e:
        print(f"DeepSeek API调用错误: {e}")
        # 发生错误时返回示例响应
        return _get_mock_response(prompt)

def _get_mock_response(prompt: str) -> str:
    """
    当无法使用真实LLM API时生成模拟响应
    
    Args:
        prompt: 输入提示
        
    Returns:
        模拟的LLM响应
    """
    # 检测提示中的关键词，生成对应的模拟响应
    if "意图" in prompt:
        # 模拟意图理解响应
        return json.dumps({
            "analysis_type": "basic_stats",
            "description": "基本统计分析",
            "target_columns": ["sales", "profit"],
            "filters": {}
        })
    elif "解释" in prompt:
        # 模拟解释生成响应
        return json.dumps({
            "summary": "这个分析显示了销售和利润的基本统计信息，帮助您了解整体业务表现。",
            "insights": [
                {
                    "title": "利润分布广泛",
                    "description": "利润的标准差较大，说明不同地区或产品类别的盈利能力存在显著差异。"
                },
                {
                    "title": "销售与利润相关",
                    "description": "从数据中可以看出，销售额较高的地区通常也有较高的利润，但这种关系并非绝对。"
                }
            ],
            "next_steps": [
                {
                    "title": "分类别分析",
                    "description": "尝试按产品类别进行分析，识别出最有利可图的产品线。"
                },
                {
                    "title": "时间趋势分析",
                    "description": "分析销售和利润随时间的变化，查看是否存在季节性模式或增长趋势。"
                }
            ]
        })
    elif "分析建议" in prompt:
        # 模拟分析建议响应
        return json.dumps([
            {
                "type": "基础统计",
                "description": "查看各列的基本统计信息",
                "value": "了解数据的整体分布和范围"
            },
            {
                "type": "比较分析",
                "description": "比较不同类别的销售额和利润",
                "value": "识别表现最好和最差的类别"
            },
            {
                "type": "时间趋势",
                "description": "分析销售额和利润的时间趋势",
                "value": "发现业务的季节性模式和增长趋势"
            }
        ])
    else:
        # 默认模拟响应
        return json.dumps({
            "response": "模拟LLM响应。在实际部署中，请提供有效的API密钥。",
            "prompt_type": "unknown"
        })

async def get_claude_response(prompt: str, model: str = "claude-3-7-sonnet-20250219") -> str:
    """使用Claude API获取LLM响应（作为备选选项）"""
    api_key = settings.CLAUDE_API_KEY
    
    if not api_key:
        # 如果API密钥未设置，返回示例响应
        return _get_mock_response(prompt)
    
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": model,
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取并返回Claude的回复
            return result['content'][0]['text']
    except Exception as e:
        print(f"LLM API调用错误: {e}")
        # 发生错误时返回示例响应
        return _get_mock_response(prompt)

async def get_openai_response(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """使用OpenAI API获取LLM响应（作为备选选项）"""
    api_key = settings.OPENAI_API_KEY
    
    if not api_key:
        # 如果API密钥未设置，返回示例响应
        return _get_mock_response(prompt)
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 800
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取并返回OpenAI的回复
            return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"OpenAI API调用错误: {e}")
        # 发生错误时返回示例响应
        return _get_mock_response(prompt)