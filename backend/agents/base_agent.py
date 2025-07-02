from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAgent(ABC):
    """所有智能体的基类，定义共通接口和功能"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理输入数据并返回结果
        
        Args:
            input_data: 输入数据
            context: 可选的上下文信息
            
        Returns:
            处理结果
        """
        pass
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        验证输入数据是否有效
        
        Args:
            input_data: 要验证的输入数据
            
        Returns:
            如果输入有效则返回True，否则返回False
        """
        return True
    
    def __str__(self) -> str:
        return f"{self.name} Agent"