from typing import Dict, Any, Optional, List
import json
import logging # Import logging
import traceback

from .base_agent import BaseAgent
from ..utils.llm_utils import get_llm_response

logger = logging.getLogger(__name__) # Initialize logger

class InterpretationAgent(BaseAgent):
    """负责将分析结果转化为通俗易懂解释的智能体"""
    
    def __init__(self):
        super().__init__("Interpretation")
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        为分析结果生成解释
        
        Args:
            input_data: 包含分析结果、可视化、意图和用户查询的字典
            context: 可选的上下文信息
            
        Returns:
            包含解释的字典
        """
        logger.info("=== InterpretationAgent 开始处理 ===")
        if not await self.validate_input(input_data):
            logger.error("InterpretationAgent 输入验证失败")
            return {"error": "Invalid input data for InterpretationAgent"}
        
        analysis_result = input_data.get("analysis_result", {})
        visualization = input_data.get("visualization", {})
        # 从 input_data 中获取 intent，如果 Orchestrator 传递了它
        # Intent 包含了 group_by 等信息，对解释 comparison 很重要
        intent = input_data.get("intent", {}) 
        query = input_data.get("query", "")
        
        logger.debug(f"InterpretationAgent received analysis_result type: {analysis_result.get('type')}")
        logger.debug(f"InterpretationAgent received intent: {intent}")

        # 使用LLM生成解释
        interpretation = await self._generate_interpretation(
            analysis_result, visualization, intent, query # 传递 intent
        )
        
        logger.info("InterpretationAgent 处理完成")
        return interpretation
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据是否包含必要的字段"""
        if "analysis_result" not in input_data:
            logger.error("InterpretationAgent: Missing 'analysis_result' in input_data")
            return False
        # query is also essential for context
        if "query" not in input_data:
            logger.warning("InterpretationAgent: Missing 'query' in input_data, LLM context might be limited.")
        # intent is highly useful
        if "intent" not in input_data:
            logger.warning("InterpretationAgent: Missing 'intent' in input_data, summary generation might be less specific.")
        return True
    
    async def _generate_interpretation(self, 
                                      analysis_result: Dict[str, Any], 
                                      visualization: Dict[str, Any], 
                                      intent: Dict[str, Any], # 接收 intent
                                      query: str) -> Dict[str, Any]:
        """使用LLM生成分析结果的解释"""
        analysis_type = analysis_result.get("type", "unknown_type")
        # result_data 是 AnalysisExecutionAgent 执行后的具体数据结果
        result_data = analysis_result.get("data", {}) 
        # metadata 是 AnalysisExecutionAgent 添加的关于分析过程的元数据
        metadata = analysis_result.get("metadata", {}) 
        
        # 准备用于LLM的分析结果摘要
        # 现在 _prepare_result_summary 会更侧重于解读 analysis_result.data
        result_summary = self._prepare_result_summary(analysis_type, result_data, metadata, intent) # 传递 intent
        
        # 准备用于LLM的可视化描述
        viz_description = self._prepare_visualization_description(visualization)
        
        # 构造LLM提示
        prompt = f"""
        你是一个数据分析助手，负责向非技术用户解释数据分析结果。
        用户的原始查询是："{query}"
        执行的分析类型是："{analysis_type}"
        分析背后的意图包括：{json.dumps(intent, ensure_ascii=False, indent=2)}

        数据分析的关键发现总结如下：
        {result_summary}

        生成了以下可视化图表：
        {viz_description}

        请用清晰、通俗易懂的中文为用户提供解释。请将您的回复结构化为包含以下键的JSON对象：
        "summary": 主要发现的简要整体总结（1-2句话）。
        "insights": 2-4个关键洞察的列表。每个洞察应该是一个包含"title"和"description"的对象。重点关注最重要或最有趣的模式、趋势、比较和变量相关性评估。
        "next_steps": 1-2个建议的后续分析步骤或用户可能考虑的问题列表。每个后续步骤应该是一个包含"title"和"description"的对象。

        您的解释应该：
        - 避免技术术语。
        - 解释数据中的模式、趋势或比较。
        - 如果适用，重点关注这些发现在实际或商业环境中可能意味着什么。
        - 如果有相关性低或无关的变量，解释原因并建议替代方案。
        - 简洁明了，所有部分总计最好在400字以内。
        - 确保JSON格式有效。
        - 必须用中文回复。
        """
        
        logger.debug(f"InterpretationAgent: Prompt for LLM (first 500 chars):\n{prompt[:500]}")

        try:
            llm_response = await get_llm_response(prompt)
            logger.debug(f"InterpretationAgent: LLM raw response:\n{llm_response}")
            interpretation = json.loads(llm_response)
            
            # Basic validation of the LLM response structure
            if not all(k in interpretation for k in ["summary", "insights", "next_steps"]):
                logger.error(f"LLM interpretation response missing required keys. Response: {interpretation}")
                raise json.JSONDecodeError("Missing required keys in LLM response", llm_response, 0)
            if not isinstance(interpretation["insights"], list) or not isinstance(interpretation["next_steps"], list):
                logger.error(f"LLM interpretation 'insights' or 'next_steps' is not a list. Response: {interpretation}")
                raise json.JSONDecodeError("'insights' or 'next_steps' not a list", llm_response, 0)

            return interpretation
        except json.JSONDecodeError as e:
            logger.error(f"InterpretationAgent: JSON parsing error from LLM response: {e}. Response snippet: {llm_response[:500] if llm_response else 'Empty Response'}")
            # Fallback to a simpler interpretation
            return {
                "summary": f"针对您的查询'{query}'的{analysis_type}分析已完成。请查看图表了解详细信息。",
                "insights": [
                    {
                        "title": "数据处理完成",
                        "description": f"分析涉及{metadata.get('filtered_rows', '多个')}个数据点，与'{', '.join(metadata.get('target_columns', ['各个方面']))}'相关。"
                    }
                ],
                "next_steps": [
                    {
                        "title": "进一步探索",
                        "description": "您可以尝试细化查询或探索数据的其他方面。"
                    }
                ]
            }
        except Exception as e:
            logger.error(f"InterpretationAgent: Error generating interpretation: {e}")
            logger.error(traceback.format_exc())
            return {
                "summary": "生成详细解释时发生错误。请参考数据和图表。",
                "insights": [],
                "next_steps": []
            }

    def _prepare_result_summary(self, 
                                analysis_type: str, 
                                data: Dict[str, Any], # This is analysis_result.data
                                metadata: Dict[str, Any], # This is analysis_result.metadata
                                intent: Dict[str, Any]) -> str: # Intent is crucial for context
        """
        准备用于LLM的分析结果摘要，更侧重于解读当前分析的具体数据。
        (Prepares a summary of the analysis results for the LLM, focusing more on the specific data from the current analysis.)
        """
        summary_parts = []
        logger.info(f"Preparing result summary for analysis type: {analysis_type}")
        
        # General metadata useful for context
        summary_parts.append(f"Analysis Type: {analysis_type}")
        summary_parts.append(f"Description from intent: {intent.get('description', 'N/A')}")
        if metadata.get('target_columns'):
            summary_parts.append(f"Target columns analyzed: {', '.join(metadata['target_columns'])}")
        if metadata.get('filtered_rows') is not None and metadata.get('original_rows') is not None:
             summary_parts.append(f"Data points analyzed: {metadata['filtered_rows']} (filtered from {metadata['original_rows']})")


        if analysis_type == "basic_stats":
            summary_parts.append("\nKey Statistics per Column:")
            for col_name, stats in data.items():
                if isinstance(stats, dict):
                    stat_desc = f"- For column '{col_name}': "
                    if "mean" in stats and stats["mean"] is not None: # Numeric column
                        # 安全地格式化数字
                        mean = stats.get('mean', 0)
                        median = stats.get('median', 0)
                        min_val = stats.get('min', 0)
                        max_val = stats.get('max', 0)
                        std_val = stats.get('std', 0)
                        
                        # 安全地格式化数值，避免f-string格式化错误
                        mean_str = f"{mean:.2f}" if isinstance(mean, (int, float)) else str(mean)
                        median_str = f"{median:.2f}" if isinstance(median, (int, float)) else str(median)
                        min_str = f"{min_val:.2f}" if isinstance(min_val, (int, float)) else str(min_val)
                        max_str = f"{max_val:.2f}" if isinstance(max_val, (int, float)) else str(max_val)
                        std_str = f"{std_val:.2f}" if isinstance(std_val, (int, float)) else str(std_val)
                        
                        stat_desc += (f"Mean={mean_str}, "
                                      f"Median={median_str}, "
                                      f"Min={min_str}, "
                                      f"Max={max_str}, "
                                      f"StdDev={std_str}, "
                                      f"Count={stats.get('count', 0)}, "
                                      f"Missing={stats.get('missing_count',0)}.")
                    elif "unique_values" in stats: # Categorical/other column
                        top_vals_str = ", ".join([f"{k}({v})" for k, v in stats.get('top_values', {}).items()][:3]) # Show top 3
                        stat_desc += (f"UniqueValues={stats['unique_values']}, TopValues=[{top_vals_str if top_vals_str else 'N/A'}], "
                                      f"Count={stats.get('count',0)}, Missing={stats.get('missing_count',0)}.")
                    else:
                        stat_desc += "No detailed stats available."
                    summary_parts.append(stat_desc)
                elif isinstance(stats, dict) and "error" in stats:
                     summary_parts.append(f"- For column '{col_name}': Error during analysis - {stats['error']}")


        elif analysis_type == "comparison":
            # group_by_cols should be in metadata or intent
            group_by_cols = metadata.get("group_by", intent.get("group_by", []))
            if not group_by_cols: # Fallback if somehow missing
                group_by_cols = ["Unknown Grouping Column"]
            summary_parts.append(f"\nComparison Results Grouped by: {', '.join(group_by_cols)}")

            for target_col, comparison_data_list in data.items():
                if isinstance(comparison_data_list, list) and comparison_data_list:
                    summary_parts.append(f"\n  Comparison for '{target_col}':")
                    # Display a few key groups, e.g., top/bottom based on a metric or just first few
                    # If sorted by intent, the first few are most relevant
                    num_items_to_show = min(len(comparison_data_list), 5) # Show up to 5 groups in summary
                    
                    for i in range(num_items_to_show):
                        group_item = comparison_data_list[i]
                        if isinstance(group_item, dict):
                            group_name_parts = [str(group_item.get(gb_col, f"N/A_for_{gb_col}")) for gb_col in group_by_cols]
                            group_name_str = ", ".join(group_name_parts)
                            
                            # Extract key metrics, ensure they exist and format
                            mean_val = group_item.get("mean")
                            median_val = group_item.get("median")
                            count_val = group_item.get("count")
                            min_val = group_item.get("min")
                            max_val = group_item.get("max")

                            # 修复f-string格式化错误：确保group_name_str安全
                            group_name_str_safe = str(group_name_str).replace('{', '{{').replace('}', '}}')
                            
                            # 安全地格式化数值
                            mean_str = f"{mean_val:.2f}" if isinstance(mean_val, (int, float)) else str(mean_val) if mean_val is not None else 'N/A'
                            median_str = f"{median_val:.2f}" if isinstance(median_val, (int, float)) else str(median_val) if median_val is not None else 'N/A'
                            min_str = f"{min_val:.2f}" if isinstance(min_val, (int, float)) else str(min_val) if min_val is not None else 'N/A'
                            max_str = f"{max_val:.2f}" if isinstance(max_val, (int, float)) else str(max_val) if max_val is not None else 'N/A'
                            
                            item_summary = (f"    - Group '{group_name_str_safe}': "
                                            f"Mean={mean_str}, "
                                            f"Median={median_str}, "
                                            f"Count={count_val if count_val is not None else 'N/A'}, "
                                            f"Min={min_str}, "
                                            f"Max={max_str}.")
                            summary_parts.append(item_summary)
                    if len(comparison_data_list) > num_items_to_show:
                        summary_parts.append(f"    ... and {len(comparison_data_list) - num_items_to_show} more groups.")
                elif isinstance(comparison_data_list, dict) and "error" in comparison_data_list:
                    summary_parts.append(f"\n  Comparison for '{target_col}': Error during analysis - {comparison_data_list['error']}")
                else:
                    summary_parts.append(f"\n  Comparison for '{target_col}': No data or unexpected data format.")
        
        elif analysis_type == "trend":
            time_col = metadata.get("time_column", "Time")
            freq = metadata.get("frequency", "Unknown Frequency")
            summary_parts.append(f"\nTrend Analysis (Time Column: '{time_col}', Frequency: '{freq}')")
            
            for target_col, trend_data_list in data.items():
                if isinstance(trend_data_list, list) and trend_data_list:
                    summary_parts.append(f"\n  Trend for '{target_col}':")
                    # Show first and last few points to give a sense of the trend
                    points_to_show_each_end = 2
                    if len(trend_data_list) <= points_to_show_each_end * 2:
                        points_to_display = trend_data_list
                    else:
                        points_to_display = trend_data_list[:points_to_show_each_end] + \
                                            [{"time": "...", "mean": "...", "count": "..."}] + \
                                            trend_data_list[-points_to_show_each_end:]
                    
                    for point in points_to_display:
                        if isinstance(point, dict):
                            time_val = point.get('time', 'N/A')
                            mean_val = point.get('mean')
                            count_val = point.get('count')
                            # 修复f-string格式化错误：分别处理数字和非数字的格式化
                            time_val_safe = str(time_val).replace('{', '{{').replace('}', '}}')
                            
                            # 安全地格式化mean_val
                            if isinstance(mean_val, (int, float)):
                                mean_str = f"{mean_val:.2f}"
                            else:
                                mean_str = str(mean_val) if mean_val is not None else 'N/A'
                            
                            item_summary = (f"    - At {time_val_safe}: "
                                            f"Mean Value={mean_str}, "
                                            f"Count={count_val if count_val is not None else 'N/A'}")
                            summary_parts.append(item_summary)
                elif isinstance(trend_data_list, dict) and "error" in trend_data_list:
                     summary_parts.append(f"\n  Trend for '{target_col}': Error during analysis - {trend_data_list['error']}")
                else:
                    summary_parts.append(f"\n  Trend for '{target_col}': No data or unexpected data format.")

        elif analysis_type == "correlation":
            summary_parts.append("\nCorrelation Analysis:")
            # data for correlation might be {"matrix_pearson": {...}, "list_pearson": [...]}
            corr_list = data.get("list_pearson", [])
            if corr_list:
                summary_parts.append("  Key Pairwise Pearson Correlations (showing strong ones |r| > 0.5 or top few):")
                strong_corrs_shown = 0
                for corr_item in sorted(corr_list, key=lambda x: abs(x.get("correlation", 0)), reverse=True):
                    if strong_corrs_shown >= 5 and abs(corr_item.get("correlation",0)) < 0.5 : # Limit number of weak correlations shown
                        if strong_corrs_shown == 5 : summary_parts.append("    ...") # Indicate more exist
                        strong_corrs_shown +=1 # ensure we don't print "..." multiple times
                        continue
                    if corr_item.get("column1") != corr_item.get("column2"): # Exclude self-correlation
                        summary_parts.append(f"    - '{corr_item.get('column1')}' and '{corr_item.get('column2')}': {corr_item.get('correlation', 0):.3f}")
                        strong_corrs_shown += 1
                        if strong_corrs_shown >= 10: # Max 10 correlations in summary
                            summary_parts.append("    ... (more correlations exist)")
                            break
                if strong_corrs_shown == 0:
                     summary_parts.append("  No strong pairwise correlations (|r| > 0.5) found among the analyzed columns.")
            elif data.get("error"):
                 summary_parts.append(f"  Error during correlation analysis: {data['error']}")
            else:
                summary_parts.append("  No correlation data available.")


        elif analysis_type == "distribution":
            summary_parts.append("\nDistribution Analysis per Column:")
            for col_name, dist_data in data.items():
                if isinstance(dist_data, dict) and "type" in dist_data:
                    summary_parts.append(f"\n  Distribution for '{col_name}' (Type: {dist_data['type']}):")
                    if dist_data["type"] == "numeric" and "stats" in dist_data:
                        stats = dist_data["stats"]
                        # 安全地格式化统计数据
                        mean = stats.get('mean', 0)
                        median = stats.get('median', 0)
                        std = stats.get('std', 0)
                        skewness = stats.get('skewness', 0)
                        kurtosis = stats.get('kurtosis', 0)
                        
                        mean_str = f"{mean:.2f}" if isinstance(mean, (int, float)) else str(mean)
                        median_str = f"{median:.2f}" if isinstance(median, (int, float)) else str(median)
                        std_str = f"{std:.2f}" if isinstance(std, (int, float)) else str(std)
                        skew_str = f"{skewness:.2f}" if isinstance(skewness, (int, float)) else str(skewness)
                        kurt_str = f"{kurtosis:.2f}" if isinstance(kurtosis, (int, float)) else str(kurtosis)
                        
                        summary_parts.append(f"    - Stats: Mean={mean_str}, Median={median_str}, StdDev={std_str}, Skew={skew_str}, Kurtosis={kurt_str}")
                        # Could summarize histogram shape if complex logic is added, or let LLM infer from raw histogram data if provided
                    elif dist_data["type"] == "categorical" and "value_counts" in dist_data:
                        top_cat_vals = {k:v for i, (k,v) in enumerate(dist_data['value_counts'].items()) if i < 3} # Top 3
                        summary_parts.append(f"    - Unique Values={dist_data.get('unique_count_with_na', 'N/A')}, Top Categories: {json.dumps(top_cat_vals, ensure_ascii=False)}")
                elif isinstance(dist_data, dict) and "error" in dist_data:
                    summary_parts.append(f"\n  Distribution for '{col_name}': Error - {dist_data['error']}")
                else:
                    summary_parts.append(f"\n  Distribution for '{col_name}': No detailed distribution data.")
        
        elif analysis_type == "prediction":
            summary_parts.append(f"\nPrediction Analysis (Time Column: {metadata.get('time_column', 'Time')})")
            for target_col, pred_data in data.items():
                if isinstance(pred_data, dict) and "forecast" in pred_data:
                    summary_parts.append(f"\n  Predictions for '{target_col}':")
                    model_details = pred_data.get("model_details", {})
                    summary_parts.append(f"    - Model Type: {model_details.get('type', 'Unknown')}")
                    if "r_squared" in model_details:
                         summary_parts.append(f"    - R-squared (Goodness of fit for historical): {model_details['r_squared']:.3f}")
                    forecast_points = pred_data.get("forecast", [])
                    if forecast_points:
                        summary_parts.append("    - Forecasted points (showing first and last if many):")
                        points_to_show = forecast_points[:1]
                        if len(forecast_points) > 2:
                            points_to_show.append({"date": "...", "forecast": "..."})
                            points_to_show.append(forecast_points[-1])
                        elif len(forecast_points) == 2:
                             points_to_show.append(forecast_points[-1])
                        
                        for fp in points_to_show:
                            forecast_val = fp.get('forecast', 'N/A')
                            # 安全地格式化预测值
                            if isinstance(forecast_val, (int, float)):
                                forecast_str = f"{forecast_val:.2f}"
                            else:
                                forecast_str = str(forecast_val) if forecast_val != 'N/A' else 'N/A'
                            
                            summary_parts.append(f"      - Date: {fp.get('date')}, Forecasted Value: {forecast_str}")
                elif isinstance(pred_data, dict) and "error" in pred_data:
                     summary_parts.append(f"\n  Predictions for '{target_col}': Error - {pred_data['error']}")
                else:
                     summary_parts.append(f"\n  Predictions for '{target_col}': No prediction data.")


        elif analysis_type == "linear_regression":
            summary_parts.append("\nLinear Regression Analysis:")
            
            # 模型信息
            model_info = data.get("model_info", {})
            if model_info:
                target_col = model_info.get("target_column", "Unknown")
                feature_cols = model_info.get("feature_columns", [])
                intercept = model_info.get("intercept", 0)
                
                summary_parts.append(f"  Target Variable: {target_col}")
                summary_parts.append(f"  Feature Variables: {', '.join(feature_cols)}")
                summary_parts.append(f"  Intercept: {intercept:.4f}")
            
            # 性能指标
            performance = data.get("performance_metrics", {})
            if performance:
                r2 = performance.get("r2", 0)
                rmse = performance.get("rmse", 0)
                mae = performance.get("mae", 0)
                
                summary_parts.append(f"  Model Performance:")
                summary_parts.append(f"    - R² (解释方差比例): {r2:.4f}")
                summary_parts.append(f"    - RMSE (均方根误差): {rmse:.4f}")
                summary_parts.append(f"    - MAE (平均绝对误差): {mae:.4f}")
            
            # 变量适用性评估
            variable_assessment = data.get("variable_assessment", {})
            if variable_assessment:
                summary_parts.append(f"  Variable Relevance Assessment:")
                
                # 高相关性变量
                high_relevance = variable_assessment.get("high_relevance_variables", [])
                if high_relevance:
                    summary_parts.append(f"    - High Relevance Variables: {', '.join(high_relevance)}")
                
                # 中等相关性变量
                medium_relevance = variable_assessment.get("medium_relevance_variables", [])
                if medium_relevance:
                    summary_parts.append(f"    - Medium Relevance Variables: {', '.join(medium_relevance)}")
                
                # 低相关性变量
                low_relevance = variable_assessment.get("low_relevance_variables", [])
                if low_relevance:
                    summary_parts.append(f"    - Low Relevance Variables: {', '.join(low_relevance)}")
                
                # 无关变量
                irrelevant_vars = variable_assessment.get("irrelevant_variables", [])
                if irrelevant_vars:
                    summary_parts.append(f"    - Irrelevant Variables: {', '.join(irrelevant_vars)}")
            
            # 特征重要性详情
            feature_importance = model_info.get("feature_importance", {})
            if feature_importance:
                summary_parts.append(f"  Feature Coefficients (including categorical dummy variables):")
                
                # 获取编码信息
                encoding_info = model_info.get("encoding_info", {})
                categorical_encoding = encoding_info.get("categorical_encoding", {})
                
                # 按特征类型分组显示
                numerical_features = []
                categorical_features = {}
                interaction_features = []
                
                for feature, info in feature_importance.items():
                    feature_type = info.get("feature_type", "numerical")
                    original_feature = info.get("original_feature", feature)
                    
                    if feature_type == "numerical":
                        numerical_features.append((feature, info))
                    elif feature_type == "categorical_dummy":
                        if original_feature not in categorical_features:
                            categorical_features[original_feature] = []
                        categorical_features[original_feature].append((feature, info))
                    elif feature_type == "interaction":
                        interaction_features.append((feature, info))
                
                # 显示数值变量
                if numerical_features:
                    summary_parts.append(f"    数值变量:")
                    for feature, info in numerical_features:
                        coefficient = info.get("coefficient", 0)
                        p_value = info.get("p_value", 1)
                        is_significant = info.get("is_significant", False)
                        significance_mark = " ***" if p_value < 0.001 else " **" if p_value < 0.01 else " *" if p_value < 0.05 else ""
                        
                        summary_parts.append(f"      {feature}: 系数 = {coefficient:.4f}, p值 = {p_value:.4f}{significance_mark}")
                
                # 显示分类变量（按原始变量分组）
                if categorical_features:
                    summary_parts.append(f"    分类变量:")
                    for original_cat, dummy_list in categorical_features.items():
                        # 获取参考类别信息
                        reference_category = "Unknown"
                        if original_cat in categorical_encoding:
                            reference_category = categorical_encoding[original_cat].get("reference_category", "Unknown")
                        
                        summary_parts.append(f"      {original_cat} (参考类别: {reference_category}):")
                        
                        for dummy_feature, info in dummy_list:
                            coefficient = info.get("coefficient", 0)
                            p_value = info.get("p_value", 1)
                            significance_mark = " ***" if p_value < 0.001 else " **" if p_value < 0.01 else " *" if p_value < 0.05 else ""
                            
                            # 提取类别名称（去掉前缀）
                            category_name = dummy_feature.replace(f"{original_cat}_", "")
                            summary_parts.append(f"        {category_name}: 系数 = {coefficient:.4f}, p值 = {p_value:.4f}{significance_mark}")
                
                # 显示交互项
                if interaction_features:
                    summary_parts.append(f"    交互项:")
                    for feature, info in interaction_features:
                        coefficient = info.get("coefficient", 0)
                        p_value = info.get("p_value", 1)
                        significance_mark = " ***" if p_value < 0.001 else " **" if p_value < 0.01 else " *" if p_value < 0.05 else ""
                        
                        summary_parts.append(f"      {feature}: 系数 = {coefficient:.4f}, p值 = {p_value:.4f}{significance_mark}")
                
                # 添加显著性说明
                summary_parts.append(f"    显著性标记: *** p<0.001, ** p<0.01, * p<0.05")
        
        else: # Fallback for unknown or custom types
            summary_parts.append(f"\nAnalysis results for '{analysis_type}':")
            try:
                # Try to pretty print a snippet of the data if it's a dict or list
                if isinstance(data, (dict, list)):
                    data_snippet = json.dumps(data, indent=2, ensure_ascii=False, default=str) # default=str for non-serializable
                    if len(data_snippet) > 500:
                        data_snippet = data_snippet[:500] + "\n... (data snippet truncated)"
                    summary_parts.append(data_snippet)
                else:
                    summary_parts.append(str(data))
            except Exception as e:
                logger.warning(f"Could not serialize data for summary for type {analysis_type}: {e}")
                summary_parts.append("Data is complex and not summarized here.")


        final_summary = "\n".join(summary_parts)
        logger.debug(f"InterpretationAgent: Final prepared summary for LLM (first 500 chars):\n{final_summary[:500]}")
        return final_summary
    
    def _prepare_visualization_description(self, visualization: Dict[str, Any]) -> str:
        """准备用于LLM的可视化描述"""
        charts = visualization.get("charts", [])
        
        if not charts:
            return "No visualizations were generated for this analysis."
        
        descriptions = ["The following visualizations are available:"]
        for i, chart_config in enumerate(charts, 1):
            chart_type = chart_config.get("type", "Unknown chart type")
            title = chart_config.get("title", f"Chart {i}")
            x_field = chart_config.get("xField")
            y_field = chart_config.get("yField")
            series_field = chart_config.get("seriesField")
            
            desc = f"- Chart {i}: A '{chart_type}' titled '{title}'."
            if x_field and y_field:
                desc += f" It shows '{y_field}' against '{x_field}'"
                if series_field:
                    desc += f", broken down by '{series_field}'."
                else:
                    desc += "."
            descriptions.append(desc)
        
        return "\n".join(descriptions)

