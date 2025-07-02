from typing import Dict, Any, Optional, List
import json

from .base_agent import BaseAgent
from ..utils.llm_utils import get_llm_response

class IntentUnderstandingAgent(BaseAgent):
    """负责理解用户自然语言查询意图的智能体 - 专注于线性回归分析"""
    
    def __init__(self):
        super().__init__("IntentUnderstanding")
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析用户查询，识别线性回归分析意图
        
        Args:
            input_data: 包含用户查询和数据分析信息的字典
            context: 可选的上下文信息
            
        Returns:
            包含结构化意图的字典
        """
        if not await self.validate_input(input_data):
            return {"error": "Invalid input data"}
        
        query = input_data.get("query", "")
        data_analysis = input_data.get("data_analysis", {})
        query_history = input_data.get("query_history", [])
        
        # 使用LLM分析查询意图
        intent = await self._analyze_intent(query, data_analysis, query_history)
        
        return intent
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据是否包含必要的字段"""
        return "query" in input_data and isinstance(input_data["query"], str)
    
    async def _analyze_intent(self, 
                             query: str, 
                             data_analysis: Dict[str, Any], 
                             query_history: List[str]) -> Dict[str, Any]:
        """使用LLM分析用户查询意图"""
        
        # 准备数据摘要用于LLM分析
        basic_info = data_analysis.get("basic_info", {})
        column_analysis = data_analysis.get("column_analysis", {})
        
        # 构建列信息摘要
        columns_info = []
        categorical_columns = []
        numerical_columns = []
        
        for col_name, col_details in column_analysis.items():
            col_type = col_details.get("type", "unknown")
            semantic_type = col_details.get("llm_semantic_type", "")
            unique_values = col_details.get("unique_values", 0)
            
            # 识别分类变量和数值变量
            if (col_type == 'object' or 
                'Categorical' in semantic_type or 
                (col_details.get("unique_values", 100) < 20 and col_type != 'float64' and col_type != 'int64')):
                categorical_columns.append(col_name)
                col_info = f"'{col_name}' (分类变量, {unique_values}个类别)"
            elif col_type in ['int64', 'float64'] or 'Numerical' in semantic_type:
                numerical_columns.append(col_name)
                col_info = f"'{col_name}' (数值变量)"
            else:
                col_info = f"'{col_name}' ({col_type})"
            
            if semantic_type:
                col_info += f" - LLM推断: {semantic_type}"
            
            columns_info.append(col_info)
        
        columns_summary = "\n".join(columns_info)
        
        # 构建查询历史摘要
        history_summary = ""
        if query_history:
            recent_queries = query_history[-3:]  # 最近3个查询
            history_summary = f"最近查询历史: {'; '.join(recent_queries)}"

        prompt = f"""
        你是一个数据分析专家，需要理解用户的自然语言查询意图，并将其转换为结构化的分析参数。

        ## 数据集信息
        数据集基本信息: {basic_info.get('rows', 0)}行, {basic_info.get('columns', 0)}列
        
        可用列信息:
        {columns_summary}
        
        分类变量列表: {categorical_columns}
        数值变量列表: {numerical_columns}
        
        {history_summary}

        ## 用户查询
        "{query}"

        ## 分析指导原则

        ### 变量选择原则
        1. **目标变量识别**: 寻找用户想要预测、解释或分析的变量（通常是因变量）
        2. **特征变量识别**: 寻找用户认为可能影响目标变量的因素（自变量）
        3. **必须包含用户明确提到的所有变量**，即使某些变量可能不相关
        4. **支持混合变量类型**: 可以同时包含数值变量和分类变量

        ### 交互效应处理原则
        1. **默认考虑交互效应**: 当同时存在数值变量和分类变量时，默认考虑它们之间的交互作用
        2. **用户明确指定**: 如果用户明确提到"交互"、"相互作用"、"不同组别的影响"等，必须考虑交互效应
        3. **用户明确排除**: 只有当用户明确说"不考虑交互"、"只要主效应"、"简单模型"时，才不考虑交互效应
        4. **交互效应类型**: 主要考虑分类变量与数值变量之间的交互

        ### 变量相关性评估
        对每个特征变量评估其与分析目标的相关性：
        - **high**: 理论上高度相关，统计上可能显著
        - **medium**: 理论上中等相关，需要统计验证  
        - **low**: 理论上弱相关，但仍有分析价值
        - **irrelevant**: 明显无关（如ID、序号等），但仍需包含在分析中

        请返回JSON格式的分析结果：
        {{
            "analysis_type": "linear_regression",
            "description": "对用户查询的简洁描述",
            "target_column": "目标变量列名",
            "feature_columns": ["特征变量1", "特征变量2", ...],
            "categorical_features": ["分类变量1", "分类变量2", ...],
            "numerical_features": ["数值变量1", "数值变量2", ...],
            "interaction_analysis": {{
                "include_interactions": true/false,
                "user_specified": "用户是否明确提到交互效应",
                "auto_detected": "是否自动检测到需要交互效应",
                "interaction_pairs": [
                    {{"categorical": "分类变量", "numerical": "数值变量", "reason": "交互原因"}}
                ],
                "reasoning": "是否考虑交互效应的原因"
            }},
            "feature_assessment": {{
                "特征变量列名1": {{
                    "relevance": "high|medium|low|irrelevant",
                    "reason": "适用性评估原因",
                    "suggestion": "建议处理方式"
                }},
                "特征变量列名2": {{
                    "relevance": "high|medium|low|irrelevant", 
                    "reason": "适用性评估原因",
                    "suggestion": "建议处理方式"
                }}
            }},
            "filters": {{
                "列名": {{
                    "operator": "equals|not_equals|greater_than|less_than|greater_equals|less_equals|in|between",
                    "value": "值或值列表"
                }}
            }},
            "regression_type": "simple|multiple",
            "expected_output": "期望的分析结果类型",
            "original_query": "原始查询"
        }}

        ## 筛选操作符使用规范
        请严格使用以下标准操作符（区分大小写）：
        - "equals": 精确匹配
        - "not_equals": 不等于匹配
        - "greater_than": 大于（数值比较）
        - "less_than": 小于（数值比较）
        - "greater_equals": 大于等于（数值比较）
        - "less_equals": 小于等于（数值比较）
        - "in": 包含于列表中（value应为数组）
        - "between": 范围匹配（value应为[min, max]数组）

        ## 重要提醒
        1. 充分利用数据集的LLM洞察，特别是数值型变量的语义类型
        2. 当用户查询模糊时，基于数据特征进行智能推断
        3. **必须包含用户明确提到的所有变量**：即使变量看起来无关，也要包含在feature_columns中
        4. 对于可能无关的变量，在feature_assessment中标记为irrelevant，但仍要包含在分析中
        5. **默认考虑交互效应**：除非用户明确说不要，否则在有分类变量时都考虑交互
        6. 确保返回的JSON格式正确且逻辑一致
        7. **必须使用标准操作符**：在filters中只能使用上述定义的操作符名称
        8. 如果无法识别合适的变量，将target_column和feature_columns设为null，让系统自动选择
        """
        
        try:
            # 调用LLM获取意图分析
            llm_response = await get_llm_response(prompt)
            
            # 解析JSON响应
            intent = json.loads(llm_response)
            
            # 添加原始查询
            intent["original_query"] = query
            
            # 数据验证和智能修正
            intent = self._validate_and_enrich_intent(intent, data_analysis)
            
            return intent
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            # 返回智能后备方案
            return self._create_fallback_intent(query, data_analysis)
        except Exception as e:
            print(f"意图分析错误: {e}")
            return {
                "error": str(e),
                "original_query": query,
                "analysis_type": "linear_regression"
            }

    def _prepare_rich_columns_info(self, data_analysis: Dict[str, Any]) -> str:
        """准备丰富的列信息，包含LLM语义分析结果"""
        if not data_analysis or "column_analysis" not in data_analysis:
            return "数据列信息不可用"
        
        columns_info = "### 可用数据列（含语义分析）:\n"
        column_analysis = data_analysis.get("column_analysis", {})
        
        for col_name, col_info in column_analysis.items():
            basic_type = col_info.get("type", "未知")
            semantic_type = col_info.get("llm_semantic_type", "Unknown")
            unique_count = col_info.get("unique_values", "N/A")
            missing_pct = col_info.get("missing_percentage", 0)
            
            col_desc = f"- **{col_name}**: {basic_type} → {semantic_type}"
            col_desc += f" (唯一值: {unique_count}, 缺失: {missing_pct:.1f}%)"
            
            # 添加LLM推断的属性
            if col_info.get("llm_inferred_properties"):
                properties = col_info["llm_inferred_properties"][:2]  # 取前2个
                col_desc += f" [{', '.join(properties)}]"
            
            columns_info += col_desc + "\n"
        
        return columns_info

    def _prepare_dataset_insights(self, data_analysis: Dict[str, Any]) -> str:
        """准备数据集整体洞察信息"""
        overall_insights = data_analysis.get("overall_llm_dataset_insights", {})
        
        if not overall_insights or overall_insights.get("error_message"):
            return "数据集整体洞察信息不可用"
        
        insights_text = "### 数据集整体洞察:\n"
        
        # 数据集类型和用途
        dataset_type = overall_insights.get("dataset_type", "未知")
        primary_purpose = overall_insights.get("primary_purpose", "未知")
        insights_text += f"- **数据集类型**: {dataset_type}\n"
        insights_text += f"- **主要用途**: {primary_purpose}\n"
        
        # 关键变量识别
        key_variables = overall_insights.get("key_variables", [])
        if key_variables:
            insights_text += f"- **关键变量**: {', '.join(key_variables)}\n"
        
        # 潜在的时间列
        time_columns = overall_insights.get("potential_time_columns", [])
        if time_columns:
            insights_text += f"- **时间列**: {', '.join([col.get('column_name', '') for col in time_columns])}\n"
        
        # 数据质量评估
        quality_assessment = overall_insights.get("data_quality_assessment", {})
        if quality_assessment:
            insights_text += f"- **数据质量**: {quality_assessment.get('overall_quality', '未知')}\n"
        
        return insights_text

    def _validate_and_enrich_intent(self, intent: Dict[str, Any], data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """验证和丰富意图信息"""
        # 确保分析类型为线性回归
        intent["analysis_type"] = "linear_regression"
        
        # 验证目标列和特征列是否存在
        column_analysis = data_analysis.get("column_analysis", {})
        available_columns = list(column_analysis.keys())
        
        # 验证目标列
        target_column = intent.get("target_column")
        if target_column and target_column not in available_columns:
            print(f"警告: 目标列 '{target_column}' 不存在于数据中")
            intent["target_column"] = None
        
        # 验证特征列
        feature_columns = intent.get("feature_columns", [])
        if feature_columns:
            valid_features = [col for col in feature_columns if col in available_columns]
            if len(valid_features) != len(feature_columns):
                print(f"警告: 部分特征列不存在于数据中")
            intent["feature_columns"] = valid_features
        
        # 确定回归类型
        if not intent.get("regression_type"):
            if not feature_columns or len(feature_columns) == 1:
                intent["regression_type"] = "simple"
            else:
                intent["regression_type"] = "multiple"
        
        return intent

    def _create_fallback_intent(self, query: str, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建后备意图，当LLM分析失败时使用"""
        return {
            "analysis_type": "linear_regression",
            "description": f"基于查询'{query}'的线性回归分析",
            "target_column": None,  # 让系统自动选择
            "feature_columns": [],  # 让系统自动选择
            "filters": {},
            "regression_type": "multiple",
            "expected_output": "线性回归模型和性能指标",
            "original_query": query
        }