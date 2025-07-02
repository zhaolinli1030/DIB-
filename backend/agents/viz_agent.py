from typing import Dict, Any, Optional, List, Callable
import json

from .base_agent import BaseAgent

class VisualizationAgent(BaseAgent):
    """负责为分析结果生成可视化的智能体 - 专注于线性回归可视化"""
    
    def __init__(self):
        super().__init__("Visualization")
        # 注册可视化处理器 - 可扩展的设计
        self.visualization_processors = {
            "linear_regression": self._visualize_linear_regression,
        }
    
    def register_processor(self, analysis_type: str, processor_func: Callable):
        """注册新的可视化处理器 - 为未来扩展提供接口"""
        self.visualization_processors[analysis_type] = processor_func
        print(f"Registered new visualization processor: {analysis_type}")
    
    def get_available_visualizations(self) -> List[str]:
        """获取所有可用的可视化类型"""
        return list(self.visualization_processors.keys())
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        为分析结果生成可视化配置
        
        Args:
            input_data: 包含分析结果和意图的字典
            context: 可选的上下文信息
            
        Returns:
            包含可视化配置的字典
        """
        if not await self.validate_input(input_data):
            return {"error": "Invalid input data"}
        
        analysis_result = input_data.get("analysis_result", {})
        intent = input_data.get("intent", {})
        
        # 根据分析类型选择合适的可视化
        analysis_type = analysis_result.get("type", "")
        
        processor = self.visualization_processors.get(analysis_type)
        if processor:
            return processor(analysis_result, intent)
        else:
            # 默认返回一个表格可视化
            return self._visualize_default(analysis_result, intent)
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据是否包含必要的字段"""
        return "analysis_result" in input_data
    
    def _visualize_linear_regression(self, analysis_result: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        """为线性回归分析生成可视化"""
        data = analysis_result.get("data", {})
        
        if not data or "error" in data:
            return self._visualize_default(analysis_result, intent)
        
        charts = []
        
        # 获取模型信息（用于后续图表）
        model_info = data.get("model_info", {})
        target_column = model_info.get("target_column", "")
        feature_columns = model_info.get("feature_columns", [])
        
        # 0. 因变量与自变量的关系图（在预测值vs实际值之前）
        raw_data = data.get("raw_data", [])
        if raw_data and target_column and feature_columns:
            # 为每个特征变量创建一个散点图
            for feature in feature_columns:
                scatter_data = []
                is_categorical = False
                
                # 检查特征是否为分类变量
                for point in raw_data:
                    if feature in point and "actual" in point:
                        feature_value = point[feature]
                        # 检查是否为分类变量（字符串类型）
                        if isinstance(feature_value, str):
                            is_categorical = True
                        break
                
                # 为分类变量和数值变量生成不同类型的图表
                if is_categorical:
                    # 分类变量：生成箱线图数据
                    category_data = {}
                    for point in raw_data:
                        if feature in point and "actual" in point:
                            category = str(point[feature])
                            actual_value = point["actual"]
                            if category not in category_data:
                                category_data[category] = []
                            category_data[category].append(actual_value)
                    
                    # 转换为箱线图格式
                    box_data = []
                    for category, values in category_data.items():
                        if values:  # 确保有数据
                            values.sort()
                            n = len(values)
                            q1 = values[n//4] if n > 0 else 0
                            median = values[n//2] if n > 0 else 0
                            q3 = values[3*n//4] if n > 0 else 0
                            min_val = min(values) if values else 0
                            max_val = max(values) if values else 0
                            
                            box_data.append({
                                "category": category,
                                "min": round(min_val, 3),
                                "q1": round(q1, 3),
                                "median": round(median, 3),
                                "q3": round(q3, 3),
                                "max": round(max_val, 3),
                                "mean": round(sum(values) / len(values), 3) if values else 0
                            })
                    
                    if box_data:
                        chart_config = {
                            "type": "box",
                            "title": f"{target_column} 按 {feature} 分组的分布",
                            "data": box_data,
                            "xField": "category",
                            "yField": ["min", "q1", "median", "q3", "max"],
                            "color": "#52C41A",
                            "description": f"展示不同{feature}类别下{target_column}的分布情况"
                        }
                        charts.append(chart_config)
                        
                        # 为每个数值变量生成按此分类变量分组的散点图
                        for numerical_feature in feature_columns:
                            if numerical_feature != feature and numerical_feature in raw_data[0]:
                                # 检查这个特征是否为数值变量
                                is_numerical = False
                                for point in raw_data:
                                    if numerical_feature in point:
                                        if isinstance(point[numerical_feature], (int, float)):
                                            is_numerical = True
                                            break
                                
                                if is_numerical:
                                    # 按分类变量分组生成散点图数据
                                    grouped_scatter_data = {}
                                    for point in raw_data:
                                        if feature in point and numerical_feature in point and "actual" in point:
                                            category = str(point[feature])
                                            x_val = point[numerical_feature]
                                            y_val = point["actual"]
                                            
                                            if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                                                if category not in grouped_scatter_data:
                                                    grouped_scatter_data[category] = []
                                                grouped_scatter_data[category].append({
                                                    "x": round(x_val, 3) if isinstance(x_val, (int, float)) else x_val,
                                                    "y": round(y_val, 3) if isinstance(y_val, (int, float)) else y_val,
                                                    "category": category
                                                })
                                    
                                    # 为每个类别生成一个散点图
                                    colors = ["#1890FF", "#52C41A", "#FA8C16", "#F5222D", "#722ED1"]
                                    for i, (category, scatter_points) in enumerate(grouped_scatter_data.items()):
                                        if scatter_points:
                                            chart_config_grouped = {
                                                "type": "scatter",
                                                "title": f"{target_column} vs {numerical_feature} ({feature}={category})",
                                                "data": scatter_points,
                                                "xField": "x",
                                                "yField": "y",
                                                "color": colors[i % len(colors)],
                                                "description": f"展示{feature}为{category}时，{target_column}与{numerical_feature}的关系"
                                            }
                                            charts.append(chart_config_grouped)
                else:
                    # 数值变量：生成普通散点图
                    for point in raw_data:
                        if feature in point and "actual" in point:
                            feature_value = point[feature]
                            actual_value = point["actual"]
                            if isinstance(feature_value, (int, float)) and isinstance(actual_value, (int, float)):
                                scatter_data.append({
                                    "x": round(feature_value, 3),
                                    "y": round(actual_value, 3)
                                })
                    
                    if scatter_data:
                        chart_config = {
                            "type": "scatter",
                            "title": f"{target_column} vs {feature}",
                            "data": scatter_data,
                            "xField": "x",
                            "yField": "y",
                            "color": "#52C41A",
                            "description": f"展示{target_column}与{feature}之间的关系"
                        }
                        charts.append(chart_config)
        
        # 1. 预测 vs 实际值散点图
        predictions = data.get("predictions", [])
        if predictions:
            scatter_data = []
            for pred in predictions:
                scatter_data.append({
                    "actual": round(pred.get("actual", 0), 3),
                    "predicted": round(pred.get("predicted", 0), 3),
                    "residual": round(pred.get("residual", 0), 3)
                })
            
            chart_config = {
                "type": "scatter",
                "title": "预测值 vs 实际值",
                "data": scatter_data,
                "xField": "actual",
                "yField": "predicted",
                "color": "#5B8FF9",
                "description": "理想情况下，点应该落在对角线上"
            }
            charts.append(chart_config)
        
        # 2. 残差图
        if predictions:
            residual_data = []
            for i, pred in enumerate(predictions):
                residual_data.append({
                    "index": i,
                    "residual": round(pred.get("residual", 0), 3),
                    "predicted": round(pred.get("predicted", 0), 3)
                })
            
            chart_config = {
                "type": "scatter",
                "title": "残差分析",
                "data": residual_data,
                "xField": "predicted",
                "yField": "residual",
                "color": "#FF6B6B",
                "description": "残差应该随机分布在0附近，没有明显的模式"
            }
            charts.append(chart_config)
        
        # 如果没有足够的数据生成图表，返回表格
        if not charts:
            return self._visualize_default(analysis_result, intent)
        
        return {
            "charts": charts,
            "suggested_layout": "grid",
            "analysis_type": "linear_regression",
            "target_column": model_info.get("target_column", ""),
            "feature_columns": model_info.get("feature_columns", []),
            "intercept": model_info.get("intercept", 0)
        }
    
    def _visualize_default(self, analysis_result: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        """默认的可视化 - 表格形式"""
        data = analysis_result.get("data", {})
        
        # 尝试将数据转换为表格格式
        if isinstance(data, dict):
            # 如果是字典，尝试展平
            table_data = []
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        table_data.append({
                            "category": key,
                            "metric": sub_key,
                            "value": str(sub_value)
                        })
                else:
                    table_data.append({
                        "metric": key,
                        "value": str(value)
                    })
        elif isinstance(data, list):
            table_data = data
        else:
            table_data = [{"value": str(data)}]
        
        return {
            "charts": [{
                "type": "table",
                "title": "分析结果",
                "data": table_data,
                "description": "分析结果的表格展示"
            }],
            "suggested_layout": "single",
            "analysis_type": analysis_result.get("type", "unknown")
        }