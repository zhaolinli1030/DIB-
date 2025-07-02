import pandas as pd
import numpy as np
import json
import traceback
import logging
import re
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from ..utils.llm_utils import get_llm_response

# 配置日志
# logging.basicConfig(level=logging.INFO) # 通常在主应用或 uvicorn 配置中设置
logger = logging.getLogger(__name__)

def convert_numpy_types(obj: Any) -> Any:
    """将NumPy类型转换为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # 处理无穷大和NaN值
        if np.isinf(obj) or np.isnan(obj):
            return None # 或者根据需求返回特定字符串如 "NaN", "Infinity"
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp): # 处理Pandas Timestamp
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class DataUnderstandingAgent(BaseAgent):
    """负责理解和分析数据结构的智能体，并利用LLM进行深度洞察"""
    
    def __init__(self):
        super().__init__("DataUnderstanding")
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析数据并提供数据摘要、统计信息、LLM增强洞察和建议
        """
        logger.info("=== DataUnderstandingAgent 开始处理 ===")
        if not await self.validate_input(input_data):
            logger.error("DataUnderstandingAgent 输入验证失败")
            return {"error": "Invalid input data for DataUnderstandingAgent"}
        
        df: pd.DataFrame = input_data.get("dataframe")
        
        # 1. 基本数据概况 (Pandas based)
        basic_info = self._get_basic_info(df)
        logger.info(f"基本信息获取完成: {basic_info['rows']} 行, {basic_info['columns']} 列")
        
        # 2. 列数据统计分析 (Pandas based)
        # 更名为 _analyze_columns_stats 以明确其产出的是统计数据
        column_statistical_analysis = self._analyze_columns_stats(df)
        logger.info("列统计分析完成")
        
        # 3. 数据质量分析 (Pandas based)
        quality_analysis = self._analyze_data_quality(df)
        logger.info("数据质量分析完成")

        # 4. LLM 增强的数据特征理解和列属性丰富
        llm_enhanced_data = await self._get_llm_enhanced_data_characterization(
            df, # 传入原始df，以便LLM可以采样（如果需要且安全）
            basic_info,
            column_statistical_analysis, # 传入纯统计的列分析结果
            quality_analysis
        )
        logger.info("LLM增强数据特征理解完成")

        # enriched_column_analysis 是合并了统计信息和LLM推断的列信息
        enriched_column_analysis = llm_enhanced_data.get("enriched_column_analysis", column_statistical_analysis)
        # overall_llm_dataset_insights 是LLM对数据集的整体洞察
        overall_llm_dataset_insights = llm_enhanced_data.get("overall_dataset_insights", {})
        
        # 5. 使用LLM生成数据分析建议（现在可以利用更丰富的上下文）
        # 传递给建议生成的数据摘要现在包含LLM的整体洞察
        data_summary_for_suggestions = {
            "basic_info": basic_info,
            "column_analysis": enriched_column_analysis, # 使用丰富后的列分析
            "quality_analysis": quality_analysis,
            "overall_llm_dataset_insights": overall_llm_dataset_insights # 新增LLM的整体洞察
        }
        analysis_suggestions = await self.generate_analysis_suggestions(data_summary_for_suggestions)
        logger.info("分析建议生成完成")
        
        result = {
            "basic_info": basic_info,
            "column_analysis": enriched_column_analysis, # 这是最终的列分析结果，包含统计和LLM洞察
            "quality_analysis": quality_analysis,
            "overall_llm_dataset_insights": overall_llm_dataset_insights, # LLM对数据集的整体洞察
            "analysis_suggestions": analysis_suggestions
        }
        
        logger.info("DataUnderstandingAgent 处理完成")
        # 转换所有NumPy类型和Pandas Timestamp为Python原生类型以便JSON序列化
        return convert_numpy_types(result)
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据是否包含有效的DataFrame"""
        if "dataframe" not in input_data or not isinstance(input_data["dataframe"], pd.DataFrame):
            logger.error("输入数据缺少DataFrame或类型不正确")
            return False
        if input_data["dataframe"].empty:
            logger.warning("输入DataFrame为空")
            # 空DataFrame本身不一定是错误，取决于后续逻辑，但在这里可以先警告
        return True
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据的基本信息"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": float(df.memory_usage(deep=True).sum()),
            "column_names": df.columns.tolist()
        }
    
    def _analyze_columns_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """分析每列的数据类型和基本统计信息（纯Pandas统计）"""
        column_analysis = {}
        if df.empty: # 处理空DataFrame的情况
            for column in df.columns: # 即使是空DataFrame，也应该有列名
                 column_analysis[column] = {
                    "type": str(df[column].dtype),
                    "unique_values": 0,
                    "missing_values": 0,
                    "missing_percentage": 0.0
                }
            return column_analysis

        for column in df.columns:
            col_data = df[column]
            col_type = str(col_data.dtype)
            
            analysis = {
                "type": col_type,
                "unique_values": int(col_data.nunique()),
                "missing_values": int(col_data.isna().sum()),
                "missing_percentage": float((col_data.isna().sum() / len(df)) * 100 if len(df) > 0 else 0)
            }
            
            if np.issubdtype(col_data.dtype, np.number):
                non_null_data = col_data.dropna()
                if not non_null_data.empty:
                    analysis.update({
                        "min": float(non_null_data.min()),
                        "max": float(non_null_data.max()),
                        "mean": float(non_null_data.mean()),
                        "median": float(non_null_data.median()),
                        "std": float(non_null_data.std()),
                        "sum": float(non_null_data.sum()) # 添加 sum
                    })
                else:
                    analysis.update({
                        "min": None, "max": None, "mean": None,
                        "median": None, "std": None, "sum": None
                    })
            elif pd.api.types.is_datetime64_any_dtype(col_data.dtype) or self._might_be_date(col_data.dropna()):
                analysis["potential_date"] = True
                non_null_data = col_data.dropna()
                if not non_null_data.empty:
                    try:
                        # 确保转换为datetime对象以获取min/max
                        dt_series = pd.to_datetime(non_null_data, errors='coerce').dropna()
                        if not dt_series.empty:
                            analysis["time_min"] = dt_series.min() # 会是Timestamp对象
                            analysis["time_max"] = dt_series.max() # 会是Timestamp对象
                    except Exception as e:
                        logger.warning(f"无法为列 {column} 计算时间范围: {e}")


            # 对分类数据的处理 (也适用于nunique较少的object类型列)
            # 增加一个判断，即使是数字类型，如果unique值很少，也可能当做分类处理部分统计
            is_likely_categorical_numeric = np.issubdtype(col_data.dtype, np.number) and col_data.nunique() < 20
            if (col_data.dtype == 'object' or pd.api.types.is_categorical_dtype(col_data.dtype) or is_likely_categorical_numeric) and col_data.nunique() < 50 : # 放宽一些限制
                # 避免对已经是 np.number 但 nunique < 20 的列重复计算 top_values
                if not (np.issubdtype(col_data.dtype, np.number) and "mean" in analysis):
                    value_counts = col_data.value_counts().head(10) # .to_dict()
                    analysis["top_values"] = {str(k): int(v) for k, v in value_counts.items()} # 确保key是str
                    analysis["value_counts_detail"] = list(value_counts.items()) # [(value, count), ...]

            column_analysis[column] = analysis
        
        return column_analysis

    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析数据质量问题"""
        if df.empty:
            return {
                "missing_values": {"total": 0, "percentage": 0.0, "by_column": {}},
                "duplicate_rows": {"total": 0, "percentage": 0.0},
                "potential_outliers_iqr": {} # 更明确是IQR方法
            }
            
        # 缺失值分析
        total_elements = df.shape[0] * df.shape[1]
        missing_total = int(df.isna().sum().sum())
        missing_percentage = float((missing_total / total_elements) * 100) if total_elements > 0 else 0.0
        missing_by_column = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}
        
        # 重复行分析
        duplicate_rows_total = int(df.duplicated().sum())
        duplicate_percentage = float((duplicate_rows_total / df.shape[0]) * 100) if df.shape[0] > 0 else 0.0
        
        # 异常值初步检测（针对数值列，使用IQR）
        outliers_iqr = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            non_null_data = df[column].dropna()
            if len(non_null_data) > 5: # 需要一些数据点来计算IQR
                q1 = float(non_null_data.quantile(0.25))
                q3 = float(non_null_data.quantile(0.75))
                iqr_value = q3 - q1
                if iqr_value > 0:  # 只在IQR大于0时检测异常值
                    lower_bound = q1 - 1.5 * iqr_value
                    upper_bound = q3 + 1.5 * iqr_value
                    outlier_count = int(((non_null_data < lower_bound) | (non_null_data > upper_bound)).sum())
                    if outlier_count > 0:
                        outliers_iqr[column] = {
                            "count": outlier_count,
                            "percentage": float(outlier_count / len(non_null_data) * 100),
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound
                        }
        
        return {
            "missing_values": {
                "total": missing_total,
                "percentage": missing_percentage,
                "by_column": missing_by_column
            },
            "duplicate_rows": {
                "total": duplicate_rows_total,
                "percentage": duplicate_percentage
            },
            "potential_outliers_iqr": outliers_iqr
        }
    
    def _might_be_date(self, series: pd.Series) -> bool:
        """检测一个列是否可能是日期（基于内容尝试转换）"""
        if series.empty or not pd.api.types.is_string_dtype(series.dtype): # 只对字符串类型且非空的数据尝试
            return False
        
        # 尝试将前 N 个非空值转换为日期，如果成功比例高，则认为是日期
        sample_size = min(len(series), 10)
        if sample_size == 0:
            return False
            
        try:
            # 尝试多种格式，errors='coerce'会将无法转换的变为NaT
            parsed_dates = pd.to_datetime(series.head(sample_size), errors='coerce', infer_datetime_format=True)
            # 如果多数能被解析 (e.g., >70%), 则认为是日期
            if parsed_dates.notna().sum() / sample_size > 0.7:
                return True
        except Exception:
            return False
        return False

    async def _get_llm_enhanced_data_characterization(self,
                                                      df: pd.DataFrame,
                                                      basic_info: Dict,
                                                      column_stats: Dict, # 来自 _analyze_columns_stats
                                                      quality_stats: Dict) -> Dict[str, Any]:
        """使用LLM对数据统计结果进行深度解读和特征推断"""
        logger.info("=== 开始使用LLM进行数据深度特征理解 ===")
        
        column_summary_for_prompt = []
        for name, stats in column_stats.items():
            col_desc = (
                f"Column '{name}': "
                f"RawType={stats.get('type')}, "
                f"UniqueValues={stats.get('unique_values')}, "
                f"Missing={stats.get('missing_percentage', 0):.2f}%"
            )
            if "mean" in stats and stats["mean"] is not None:
                col_desc += (
                    f", Mean={stats['mean']:.2f}, Std={stats.get('std', 0):.2f}, "
                    f"Min={stats['min']:.2f}, Max={stats['max']:.2f}"
                )
            elif "top_values" in stats and stats["top_values"]:
                top_vals_str = ", ".join([f"{k}({v})" for k, v in stats['top_values'].items()])
                col_desc += f", Top10Values=[{top_vals_str}]"
            if stats.get("potential_date"):
                col_desc += f", PotentialDate=True (MinTime: {stats.get('time_min', 'N/A')}, MaxTime: {stats.get('time_max', 'N/A')})"
            column_summary_for_prompt.append(col_desc)
        
        df_sample_str = "No data sample provided for LLM."
        if not df.empty:
            try:
                # 选择性采样，避免列过多或行过长，优先选择多样化的列
                sample_df = df.head()
                if df.shape[1] > 10: # 如果列太多，取前5列和后5列，或随机选择
                    sample_df = pd.concat([df.iloc[:, :5], df.iloc[:, -5:]], axis=1).head()
                
                # 限制每列字符串长度，避免过长的单元格内容
                for col in sample_df.columns:
                    if sample_df[col].dtype == 'object':
                        sample_df[col] = sample_df[col].astype(str).str.slice(0, 50) # 每单元格最多50字符

                df_sample_str = sample_df.to_markdown(index=False, tablefmt="pipe")
                max_sample_length = 2000  # 调整Token预算
                if len(df_sample_str) > max_sample_length:
                    df_sample_str = df_sample_str[:max_sample_length] + "\n... (sample truncated due to length)"
            except Exception as e:
                logger.warning(f"生成DataFrame样本字符串时出错: {e}")
                df_sample_str = "Error generating DataFrame sample."

        prompt = f"""
        You are an expert data analyst tasked with providing deep insights into a dataset based on its statistical summary and a small data sample. Your goal is to enrich the understanding of each column and the dataset as a whole.

        Dataset Statistical Summary:
        - Rows: {basic_info['rows']}
        - Columns: {basic_info['columns']} (Column Names: {', '.join(basic_info['column_names'][:15])}{'...' if len(basic_info['column_names']) > 15 else ''})
        - Overall Missing Values: {quality_stats['missing_values']['percentage']:.2f}%
        - Duplicate Rows: {quality_stats['duplicate_rows']['percentage']:.2f}%
        - Columns with Potential Outliers (IQR method): {', '.join(quality_stats['potential_outliers_iqr'].keys()) if quality_stats['potential_outliers_iqr'] else 'None identified by IQR'}

        Detailed Column Statistics:
        {chr(10).join(column_summary_for_prompt)}

        Data Sample (first few rows, some cell values might be truncated):
        {df_sample_str}

        Based on ALL the information above (column names, statistical details, and data sample), please provide the following in a SINGLE, VALID JSON object:

        1.  "column_interpretations": An object where each key is an EXACT column name from the dataset. For each column, provide:
            * "semantic_type": Infer the most likely semantic type. Examples: "Identifier_Unique", "Identifier_NonUnique", "Categorical_LowCardinality", "Categorical_HighCardinality", "Numerical_Continuous", "Numerical_Discrete", "Datetime_Year", "Datetime_Date", "Datetime_Timestamp", "Boolean", "Text_Short", "Text_Long", "Geolocation_Coordinate", "Percentage", "Currency", "Constant", "Other_MixedTypes", "Unknown". 请用中文描述。
            * "inferred_properties": A list of 2-3 concise observations or inferred properties. Examples: "Likely primary key if unique", "Represents annual data points", "Highly skewed distribution", "Appears to be free-form user comments", "All values are positive", "Constant value, low analytical utility". 请用中文描述。
            * "data_quality_flags": A list of potential data quality issues specific to this column observed from stats or sample. Examples: "High missing rate (xx%)", "Suspected outliers", "Mixed data types apparent in sample", "Inconsistent date format in sample", "Trailing/leading spaces in sample". 请用中文描述。
            * "initial_analysis_suggestions": A list of 1-2 very specific initial analysis steps for this column. Examples: "If 'Datetime_Year', use for yearly trend analysis", "If 'Categorical_LowCardinality', use for grouping/segmentation", "If 'Text_Long', consider NLP for sentiment/topics". 请用中文描述。

        2.  "overall_dataset_insights": An object containing:
            * "dataset_description_hypothesis": A brief (1-2 sentences) hypothesis about what the dataset represents or its primary purpose, based on column names and inferred types. 请用中文描述。
            * "potential_time_columns": A list of column names that seem to represent time dimensions. For each, suggest a "likely_granularity" (e.g., "Year", "Month", "Date", "Timestamp"). Example: [{{"column_name": "年份", "likely_granularity": "Year"}}, {{"column_name": "order_date", "likely_granularity": "Date"}}]. If none, provide an empty list. 请用中文描述。
            * "potential_key_or_identifier_columns": A list of column names that are likely unique identifiers (e.g., primary keys) or important grouping/dimensional keys. 请用中文描述。
            * "general_data_quality_summary": A 1-2 sentence summary of the overall data quality based on missing values, duplicates, and outlier information. 请用中文描述。
            * "cross_column_observations": Any interesting potential relationships or redundancies observed between columns. Example: "'product_id' and 'product_name' seem to refer to the same entity". If none, empty list. 请用中文描述。
            * "overall_analysis_theme_suggestions": 2-3 high-level themes or types of analysis that seem most promising for this dataset as a whole. Examples: "Time series trend analysis", "Customer segmentation", "Product performance comparison". 请用中文描述。

        Output ONLY the valid JSON object. Do not add any explanations or text before or after the JSON.
        For "semantic_type", strictly choose from the provided examples or use "Unknown" if truly unsure.
        If a list is empty (e.g., no "data_quality_flags" for a column), use an empty list [].
        Example for a column in "column_interpretations":
        "年份": {{
            "semantic_type": "Datetime_Year",
            "inferred_properties": ["Integer values representing calendar years", "Likely primary time dimension"],
            "data_quality_flags": [],
            "initial_analysis_suggestions": ["Use for yearly trend aggregation", "Convert to YYYY-01-01 for time series functions"]
        }}

        所有输出内容（包括类型、属性、标记、建议等）必须用中文描述，不要用英文。
        """
        logger.debug(f"LLM Prompt for Data Characterization (first 500 chars):\n{prompt[:500]}")
        llm_response_str = await get_llm_response(prompt)
        # logger.debug(f"LLM Raw Response for Data Characterization:\n{llm_response_str}")

        enriched_column_analysis = column_stats.copy() # Start with original stats
        overall_dataset_insights = {"error_message": "LLM processing did not complete successfully or returned invalid format."}


        try:
            llm_insights = json.loads(llm_response_str)
            
            if not isinstance(llm_insights, dict) or \
               "column_interpretations" not in llm_insights or \
               "overall_dataset_insights" not in llm_insights:
                logger.error("LLM response for data characterization has incorrect top-level structure.")
                # Keep the error message in overall_dataset_insights
            else:
                # Process column_interpretations
                llm_column_interpretations = llm_insights.get("column_interpretations", {})
                for col_name, stats_val in enriched_column_analysis.items():
                    if col_name in llm_column_interpretations:
                        col_interp = llm_column_interpretations[col_name]
                        # Ensure col_interp is a dict before trying to get keys
                        if isinstance(col_interp, dict):
                            stats_val["llm_semantic_type"] = col_interp.get("semantic_type", "Unknown")
                            stats_val["llm_inferred_properties"] = col_interp.get("inferred_properties", [])
                            stats_val["llm_data_quality_flags"] = col_interp.get("data_quality_flags", [])
                            stats_val["llm_initial_analysis_suggestions"] = col_interp.get("initial_analysis_suggestions", [])
                        else:
                            logger.warning(f"LLM interpretation for column '{col_name}' is not a dictionary: {col_interp}")
                            stats_val["llm_error"] = "Invalid interpretation format from LLM."
                
                # Store overall_dataset_insights
                overall_dataset_insights = llm_insights.get("overall_dataset_insights", {})
                if "error_message" in overall_dataset_insights: # Clear default error if LLM provided this key
                    del overall_dataset_insights["error_message"]


        except json.JSONDecodeError as e:
            logger.error(f"LLM数据特征理解响应JSON解析失败: {e}. Response: {llm_response_str[:500]}...") # Log snippet
            # overall_dataset_insights already has an error message
            overall_dataset_insights["json_decode_error"] = str(e)
            overall_dataset_insights["raw_llm_response_snippet"] = llm_response_str[:500] + "..."
        except Exception as e:
            logger.error(f"处理LLM数据特征理解响应时发生未知错误: {e}")
            logger.error(traceback.format_exc())
            overall_dataset_insights["unknown_processing_error"] = str(e)

        return {
            "enriched_column_analysis": enriched_column_analysis,
            "overall_dataset_insights": overall_dataset_insights
        }

    async def generate_analysis_suggestions(self, data_summary_for_suggestions: Dict[str, Any]) -> List[Dict[str, str]]:
        """基于（可能）增强的数据摘要生成分析建议"""
        logger.info("=== 开始生成分析建议 (使用LLM增强信息) ===")
        # data_summary_for_suggestions now includes 'basic_info', 
        # 'column_analysis' (enriched), 'quality_analysis', 
        # and 'overall_llm_dataset_insights'.

        try:
            prompt = self._create_suggestion_prompt_with_llm_insights(data_summary_for_suggestions)
            # logger.debug(f"发送给LLM的建议生成提示词:\n{prompt}")
            
            response = await get_llm_response(prompt)
            # logger.debug(f"LLM建议原始响应: {response}")
            
            suggestions = self._process_llm_suggestion_response(response, data_summary_for_suggestions) # Renamed for clarity
            
            logger.info(f"=== 最终返回的建议 (使用LLM增强信息) ===")
            for i, suggestion in enumerate(suggestions):
                logger.info(f"建议 {i+1}: {suggestion}")
            return suggestions
                
        except Exception as e:
            logger.error(f"生成分析建议失败: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback to default suggestions, now also potentially smarter
            default_suggestions = self._get_enhanced_default_suggestions(data_summary_for_suggestions)
            logger.info(f"返回默认建议: {default_suggestions}")
            return default_suggestions

    def _create_suggestion_prompt_with_llm_insights(self, data_summary: Dict[str, Any]) -> str:
        """创建结构化的提示词以生成分析建议，现在利用LLM的整体洞察"""
        basic_info = data_summary['basic_info']
        quality_analysis = data_summary['quality_analysis']
        # column_analysis is the enriched version, containing llm_semantic_type etc.
        column_analysis_enriched = data_summary['column_analysis'] 
        overall_insights = data_summary.get('overall_llm_dataset_insights', {})

        # 提取一些关键的列信息用于Prompt，避免过长
        column_prompt_summary = []
        selected_columns_for_prompt = list(column_analysis_enriched.keys())[:7] # Limit to first 7 for prompt brevity
        for col_name in selected_columns_for_prompt:
            stats = column_analysis_enriched[col_name]
            desc = f"- Column '{col_name}': RawType={stats.get('type')}, SemanticType={stats.get('llm_semantic_type', 'N/A')}, Missing={stats.get('missing_percentage',0):.1f}%."
            if stats.get('llm_inferred_properties'):
                desc += f" Observations: {', '.join(stats['llm_inferred_properties'][:2])}." # First 2 properties
            column_prompt_summary.append(desc)
        
        overall_insights_summary = "Key LLM-derived Dataset Insights:\n"
        if overall_insights and not overall_insights.get("error_message"): # Check if insights are valid
            overall_insights_summary += f"  - Hypothesis: {overall_insights.get('dataset_description_hypothesis', 'N/A')}\n"
            if overall_insights.get('potential_time_columns'):
                time_cols_info = [f"{tc.get('column_name')} ({tc.get('likely_granularity')})" for tc in overall_insights['potential_time_columns']]
                overall_insights_summary += f"  - Potential Time Columns: {', '.join(time_cols_info)}\n"
            if overall_insights.get('potential_key_or_identifier_columns'):
                overall_insights_summary += f"  - Potential Key/ID Columns: {', '.join(overall_insights['potential_key_or_identifier_columns'])}\n"
            if overall_insights.get('general_data_quality_summary'):
                 overall_insights_summary += f"  - Data Quality Note: {overall_insights['general_data_quality_summary']}\n"
            if overall_insights.get('overall_analysis_theme_suggestions'): # These themes might be in English
                overall_insights_summary += f"  - Suggested Analysis Themes (from initial LLM pass): {', '.join(overall_insights['overall_analysis_theme_suggestions'])}\n"
        else:
            overall_insights_summary = "No specific LLM-derived dataset insights available for suggestion generation, relying on stats.\n"


        prompt = f"""
        You are an expert data analysis consultant. Your primary language for communication with the user is CHINESE (Simplified Chinese).
        Based on the following comprehensive data summary, which includes statistical facts and AI-derived interpretations (these might be in English, use them for context but ensure your final output is in Chinese), your task is to generate 4-6 specific, actionable, and insightful data analysis suggestions for a user.

        Overall Data Profile:
        - Rows: {basic_info.get('rows', 'N/A')}, Columns: {basic_info.get('columns', 'N/A')}
        - Missing Data (Overall): {quality_analysis.get('missing_values', {}).get('percentage', 0):.1f}%
        - Duplicate Rows: {quality_analysis.get('duplicate_rows', {}).get('percentage', 0):.1f}%

        Summary of Selected Columns (Statistical & AI Inferred Semantic Types/Properties - these might be in English):
        {chr(10).join(column_prompt_summary)}
        {'... (more columns exist)' if len(column_analysis_enriched) > len(selected_columns_for_prompt) else ''}

        {overall_insights_summary} 

        Critical Instructions for Generating Suggestions in CHINESE:
        1. Generate 4 to 6 suggestions.
        2. Each suggestion MUST be entirely in CHINESE (Simplified Chinese).
        3. Each suggestion must be concrete and tell the user WHAT to analyze, WHICH key columns to use (refer to column names as they appear in the data summary, e.g., '{selected_columns_for_prompt[0] if selected_columns_for_prompt else 'column_name'}'), and WHY it's valuable (the insight or business value).
        4. Leverage the "Key LLM-derived Dataset Insights" (provided above, potentially in English) for inspiration, but rephrase and present your final suggestions in fluent CHINESE.
        5. Prioritize suggestions that seem most impactful or revealing based on the data's nature.
        6. Avoid generic suggestions; make them specific to the potential of THIS dataset.

        Return your suggestions STRICTLY in the following JSON array format. Ensure all string values within the JSON are in CHINESE. Do NOT add any other text or markdown before or after the JSON object:
        [
          {{
            "type": "具体的分析标题 (例如：'年度销售趋势分析')",
            "description": "对分析的详细中文描述。提及关键列名。例如：'分析【销售额】随时间列【年份】的变化趋势，以识别增长模式。'",
            "value": "此分析的商业价值或能获得的具体洞察。例如：'识别销售高峰期并预测未来销售额。'"
          }},
          {{
            "type": "按【产品类别】分组的销售额对比",
            "description": "比较不同【产品类别】下的【销售总额】均值或总和，识别各类别的表现差异。",
            "value": "找出畅销和滞销的产品类别，为库存和营销策略提供依据。"
          }}
          // ... 您可以根据需要添加更多中文建议示例，但请确保这里的示例和最终输出都符合上述JSON结构 ...
        ]
        """
        return prompt

    def _process_llm_suggestion_response(self, response: str, context_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """处理LLM的分析建议响应（与原_process_llm_response类似，可能略作调整）"""
        if not response or not response.strip():
            logger.warning("收到空的LLM建议响应")
            return self._get_enhanced_default_suggestions(context_data)
        
        cleaned_response = self._clean_response_format(response)
        # logger.debug(f"清理后的LLM建议响应: {cleaned_response}")
        
        try:
            suggestions = json.loads(cleaned_response)
            if not isinstance(suggestions, list):
                logger.warning("LLM建议响应不是预期的数组格式")
                return self._get_enhanced_default_suggestions(context_data)
            
            validated_suggestions = []
            for suggestion in suggestions:
                if isinstance(suggestion, dict) and all(key in suggestion for key in ['type', 'description', 'value']):
                    validated_suggestions.append({
                        "type": str(suggestion['type']),
                        "description": str(suggestion['description']),
                        "value": str(suggestion['value'])
                    })
            
            if not validated_suggestions:
                logger.warning("LLM建议响应中没有找到有效的建议结构")
                return self._get_enhanced_default_suggestions(context_data)
            
            return validated_suggestions
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM建议响应JSON解析失败: {str(e)}. Response: {cleaned_response[:500]}")
            return self._get_enhanced_default_suggestions(context_data)

    def _clean_response_format(self, response: str) -> str:
        """清理LLM响应格式，提取JSON部分"""
        response = response.strip()
        
        # 移除markdown代码块
        # Pattern to find ```json ... ``` or ``` ... ```
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if match:
            response = match.group(1).strip()
        else:
            # If no markdown block, try to find first '[' and last ']' if it looks like a list
            # or first '{' and last '}' if it looks like an object, assuming response is *only* JSON.
            if response.startswith('[') and response.endswith(']'):
                pass # Looks like a JSON array already
            elif response.startswith('{') and response.endswith('}'):
                pass # Looks like a JSON object already
            else:
                # Fallback: try to find the first '[' or '{' and last ']' or '}'
                # This is heuristic and might not always be correct
                start_bracket = -1
                end_bracket = -1
                if '[' in response and ']' in response:
                    start_bracket = response.find('[')
                    end_bracket = response.rfind(']')
                elif '{' in response and '}' in response:
                    start_bracket = response.find('{')
                    end_bracket = response.rfind('}')

                if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                    response = response[start_bracket : end_bracket+1]
        
        return response

    def _get_enhanced_default_suggestions(self, data_summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """根据数据摘要（现在可能包含LLM洞察）生成增强的默认建议"""
        suggestions = []
        basic_info = data_summary.get('basic_info', {})
        quality_analysis = data_summary.get('quality_analysis', {"missing_values":{}, "duplicate_rows":{}, "potential_outliers_iqr":{}})
        overall_insights = data_summary.get('overall_llm_dataset_insights', {})

        # 基础统计分析总是好的起点
        suggestions.append({
            "type": "描述性统计与分布概览",
            "description": f"对数据集中的 {basic_info.get('columns', '所有')} 个变量进行基本统计描述（均值、中位数、极差等）并查看其数值分布或类别频次。",
            "value": "快速了解数据的中心趋势、离散程度、分布形态和主要类别构成。"
        })
        
        # 根据LLM识别的时间列推荐趋势分析
        if overall_insights and isinstance(overall_insights.get('potential_time_columns'), list):
            for time_col_info in overall_insights['potential_time_columns']:
                col_name = time_col_info.get('column_name')
                granularity = time_col_info.get('likely_granularity', '合适的时间单位')
                if col_name:
                    suggestions.append({
                        "type": f"{granularity}趋势分析 ({col_name})",
                        "description": f"分析关键数值指标随时间列 '{col_name}' (按{granularity})的变化趋势。",
                        "value": f"发现业务的周期性模式、增长趋势或异常波动点，基于{granularity}视角。"
                    })
                    break # 通常一个主要时间趋势就够作为默认建议了

        # 根据LLM识别的关键列推荐分组比较
        if overall_insights and isinstance(overall_insights.get('potential_key_or_identifier_columns'), list):
            key_cols = overall_insights['potential_key_or_identifier_columns']
            # 尝试找到一个非ID类的、适合分组的类别列
            categorical_group_col = None
            if data_summary.get("column_analysis"):
                for col_name in key_cols:
                    col_detail = data_summary["column_analysis"].get(col_name, {})
                    semantic_type = col_detail.get("llm_semantic_type", "")
                    if "Categorical" in semantic_type and col_detail.get("unique_values", 100) < 30: # 适合分组的类别列
                        categorical_group_col = col_name
                        break
            if categorical_group_col:
                 suggestions.append({
                    "type": f"按 '{categorical_group_col}' 分组的关键指标比较",
                    "description": f"比较不同 '{categorical_group_col}' 类别下，核心数值指标（如销售额、用户数等）的均值或总和。",
                    "value": f"识别在 '{categorical_group_col}'维度下表现突出或欠佳的群体，发现差异化特征。"
                })


        # 数据质量检查建议 (如果存在问题)
        if quality_analysis.get('missing_values',{}).get('percentage', 0) > 5:
            suggestions.append({
                "type": "缺失数据探查与处理",
                "description": f"数据集中存在约 {quality_analysis['missing_values']['percentage']:.1f}% 的缺失值，请重点关注缺失严重的列并考虑处理策略（如填充或删除）。",
                "value": "提高数据完整性，确保分析结果的准确性和可靠性。"
            })
        
        if quality_analysis.get('duplicate_rows',{}).get('percentage', 0) > 1:
            suggestions.append({
                "type": "重复数据审查",
                "description": f"发现数据中存在 {quality_analysis['duplicate_rows']['percentage']:.1f}% 的重复行，建议审查并考虑去重。",
                "value": "确保数据记录的唯一性，避免统计偏差。"
            })

        if len(quality_analysis.get('potential_outliers_iqr', {})) > 0:
            outlier_cols_str = ", ".join(list(quality_analysis['potential_outliers_iqr'].keys())[:3])
            suggestions.append({
                "type": "潜在异常值检测",
                "description": f"在列（如 {outlier_cols_str}）中发现了潜在的异常值，建议使用箱线图或散点图进一步确认并判断是否需要处理。",
                "value": "识别数据中的极端个案，理解其成因或排除其对整体分析的干扰。"
            })
        
        # 通用相关性分析 (如果数值列较多)
        num_numeric_cols = 0
        if data_summary.get("column_analysis"):
            for col_detail in data_summary["column_analysis"].values():
                if "Numerical" in col_detail.get("llm_semantic_type", "") or \
                   (isinstance(col_detail.get("type"), str) and ("int" in col_detail["type"] or "float" in col_detail["type"])):
                    num_numeric_cols +=1
        
        if num_numeric_cols >= 2:
            suggestions.append({
                "type": "核心数值指标相关性分析",
                "description": "计算主要数值指标之间的相关系数（如皮尔逊相关系数），并可视化相关性矩阵热力图。",
                "value": "发现变量间的线性关联强度和方向，理解哪些指标可能相互影响。"
            })
        
        return suggestions[:6] # 最多返回6个建议
