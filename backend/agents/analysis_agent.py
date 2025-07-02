import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
import traceback
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class AnalysisExecutionAgent(BaseAgent):
    """负责执行数据分析操作的智能体 - 专注于线性回归分析，支持分类变量和交互效应"""
    
    def __init__(self):
        super().__init__("AnalysisExecution")
        # 注册分析处理器 - 可扩展的设计
        self.analysis_processors = {
            "linear_regression": self._process_linear_regression,
        }
    
    def register_processor(self, analysis_type: str, processor_func: Callable):
        """注册新的分析处理器 - 为未来扩展提供接口"""
        self.analysis_processors[analysis_type] = processor_func
        logger.info(f"Registered new analysis processor: {analysis_type}")
    
    def get_available_analyses(self) -> List[str]:
        """获取所有可用的分析类型"""
        return list(self.analysis_processors.keys())
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据是否包含必要的字段"""
        logger.info(f"Validating AnalysisExecutionAgent input: {list(input_data.keys())}")
        
        if "intent" not in input_data:
            logger.error("Missing 'intent' in input_data")
            return False
        
        if "data" not in input_data:
            logger.error("Missing 'data' (DataFrame) in input_data") 
            return False
        
        df = input_data["data"]
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Invalid DataFrame type: {type(df)}")
            return False
        
        logger.info(f"Input validation passed for AnalysisExecutionAgent.")
        return True

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        根据解析的意图执行数据分析
        """
        logger.info("=== AnalysisExecutionAgent 开始处理 ===")
        logger.info(f"Input data keys: {list(input_data.keys())}")
        
        if not await self.validate_input(input_data):
            logger.error("AnalysisExecutionAgent Input validation failed")
            return {"error": "Invalid input data for AnalysisExecutionAgent"}
        
        intent = input_data.get("intent", {})
        df_original = input_data.get("data")
        if not isinstance(df_original, pd.DataFrame):
            logger.error("Data is not a DataFrame")
            return {"error": "Invalid data type"}
        data_understanding_insights: Optional[Dict[str, Any]] = input_data.get("data_analysis") 
        
        if df_original.empty:
            logger.warning("Input DataFrame is empty.")
            return {
                "error": "DataFrame is empty. Cannot perform analysis.",
                "data": {},
                "metadata": {
                    "original_rows": 0,
                    "filtered_rows": 0,
                    "analysis_type": intent.get("analysis_type", "unknown"),
                    "description": intent.get("description", ""),
                    "target_columns": intent.get("target_columns", [])
                }
            }

        logger.info(f"Intent: {intent}")
        logger.info(f"Original DataFrame shape: {df_original.shape}")
        
        # 应用过滤条件
        df = df_original.copy()
        filtered_df = self._apply_filters(df, intent.get("filters", {}))
        logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")

        if filtered_df.empty:
            logger.warning("DataFrame is empty after applying filters.")
            return {
                "error": "Data is empty after applying filters. Cannot perform analysis.",
                "data": {},
                "metadata": {
                    "original_rows": len(df_original),
                    "filtered_rows": 0,
                    "analysis_type": intent.get("analysis_type", "unknown"),
                    "description": intent.get("description", ""),
                    "target_columns": intent.get("target_columns", [])
                }
            }
        
        analysis_type = intent.get("analysis_type", "linear_regression")
        processor = self.analysis_processors.get(analysis_type)

        if not processor:
            logger.error(f"No processor found for analysis_type: {analysis_type}")
            return {
                "error": f"Analysis type '{analysis_type}' is not supported. Available types: {self.get_available_analyses()}",
                "data": {},
                "metadata": {
                    "original_rows": len(df_original),
                    "filtered_rows": len(filtered_df),
                    "analysis_type": analysis_type,
                    "description": intent.get("description", ""),
                    "target_columns": intent.get("target_columns", [])
                }
            }
        
        logger.info(f"Using processor for: {analysis_type}")
        
        # 执行分析
        analysis_result = await processor(filtered_df, intent, data_understanding_insights)
        
        # 添加元数据
        analysis_result["metadata"] = {
            "original_rows": len(df_original),
            "filtered_rows": len(filtered_df),
            "analysis_type": analysis_type,
            "description": intent.get("description", ""),
            "target_columns": intent.get("target_columns", []),
            "available_analyses": self.get_available_analyses()
        }
        
        logger.info("AnalysisExecutionAgent 处理完成")
        return analysis_result
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """应用意图中指定的过滤条件，支持智能匹配"""
        if not filters or df.empty:
            return df
        
        filtered_df = df.copy()
        
        logger.info(f"Available columns in DataFrame: {list(df.columns)}")
        logger.info(f"DataFrame shape before filtering: {df.shape}")
        
        for column, filter_spec in filters.items():
            if column not in filtered_df.columns:
                logger.warning(f"Filter column '{column}' not found in DataFrame. Available columns: {list(filtered_df.columns)}")
                continue
            
            operator = filter_spec.get("operator", "equals")
            value = filter_spec.get("value")
            
            unique_values = filtered_df[column].unique()[:10]
            logger.info(f"Column '{column}' unique values (first 10): {list(unique_values)}")
            logger.info(f"Looking for value: '{value}' with operator: '{operator}'")
            
            try:
                original_shape = filtered_df.shape
                
                if operator in ["equals", "="]:
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif operator in ["not_equals", "!="]:
                    filtered_df = filtered_df[filtered_df[column] != value]
                elif operator == "greater_than":
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif operator == "less_than":
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif operator == "greater_equals":
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == "less_equals":
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == "in":
                    if isinstance(value, list):
                        exact_matches = filtered_df[filtered_df[column].isin(value)]
                        if exact_matches.empty and filtered_df[column].dtype == 'object':
                            logger.info(f"Exact match failed for 'in' operator, attempting fuzzy matching...")
                            matched_values = self._fuzzy_match_values(filtered_df[column].unique(), value)
                            if matched_values:
                                logger.info(f"Fuzzy matching found: {matched_values}")
                                filtered_df = filtered_df[filtered_df[column].isin(matched_values)]
                            else:
                                logger.warning(f"No fuzzy matches found for values: {value}")
                                filtered_df = exact_matches
                        else:
                            filtered_df = exact_matches
                    else:
                        logger.warning(f"Filter 'in' for column '{column}' expects a list value, got {type(value)}. Skipping.")
                elif operator == "between":
                    if isinstance(value, list) and len(value) == 2:
                        filtered_df = filtered_df[(filtered_df[column] >= value[0]) & 
                                                  (filtered_df[column] <= value[1])]
                    else:
                        logger.warning(f"Filter 'between' for column '{column}' expects a list of two values, got {value}. Skipping.")
                else:
                    logger.warning(f"Unsupported filter operator '{operator}' for column '{column}'. Skipping.")
                
                logger.info(f"Filter result: {original_shape} -> {filtered_df.shape}")
                
            except Exception as e:
                logger.error(f"Error applying filter on column '{column}' with operator '{operator}' and value '{value}': {e}")
                logger.error(traceback.format_exc())
                continue 
        
        logger.info(f"Final filtered DataFrame shape: {filtered_df.shape}")
        return filtered_df
    
    def _fuzzy_match_values(self, available_values: np.ndarray, target_values: List[Any]) -> List[str]:
        """智能匹配值，支持部分匹配和常见变体"""
        import difflib
        
        matched_values = []
        
        for target in target_values:
            target_str = str(target).strip()
            
            # 1. 直接匹配
            if target_str in available_values:
                matched_values.append(target_str)
                continue
            
            # 2. 忽略大小写匹配
            target_lower = target_str.lower()
            for available in available_values:
                if str(available).lower() == target_lower:
                    matched_values.append(str(available))
                    break
            else:
                # 3. 模糊匹配
                matches = difflib.get_close_matches(target_str, [str(v) for v in available_values], n=1, cutoff=0.6)
                if matches:
                    matched_values.append(matches[0])
        
        return matched_values

    async def _process_linear_regression(self, df: pd.DataFrame, intent: Dict[str, Any], data_understanding_insights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理线性回归分析 - 支持简单和多元线性回归，包含分类变量和交互效应"""
        logger.info("=== Linear Regression Analysis with Categorical Variables ===")
        
        if df.empty:
            return {"type": "linear_regression", "data": {}, "error": "Input data is empty."}

        # 获取目标变量和特征变量
        target_column = intent.get("target_column")
        feature_columns = intent.get("feature_columns", [])
        categorical_features = intent.get("categorical_features", [])
        numerical_features = intent.get("numerical_features", [])
        interaction_analysis = intent.get("interaction_analysis", {})
        
        # 如果没有指定，尝试自动识别
        if not target_column or not feature_columns:
            auto_identified = self._auto_identify_regression_variables(df, data_understanding_insights)
            if not target_column:
                target_column = auto_identified.get("target_column")
            if not feature_columns:
                feature_columns = auto_identified.get("feature_columns", [])
        
        # 自动识别分类变量和数值变量（如果Intent中没有明确指定）
        if not categorical_features and not numerical_features:
            categorical_features, numerical_features = self._identify_variable_types(df, feature_columns)
        
        # 记录原始用户指定的变量
        logger.info(f"用户指定的目标变量: {target_column}")
        logger.info(f"用户指定的特征变量: {feature_columns}")
        logger.info(f"分类变量: {categorical_features}")
        logger.info(f"数值变量: {numerical_features}")
        logger.info(f"交互分析配置: {interaction_analysis}")
        
        if not target_column or not feature_columns:
            return {
                "type": "linear_regression",
                "error": "Could not identify target column or feature columns for regression analysis.",
                "data": {}
            }
        
        # 验证列是否存在
        if target_column not in df.columns:
            return {
                "type": "linear_regression",
                "error": f"Target column '{target_column}' not found in data.",
                "data": {}
            }
        
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            return {
                "type": "linear_regression",
                "error": f"Feature columns not found in data: {missing_features}",
                "data": {}
            }

        # 准备数据
        try:
            # 选择相关列并删除缺失值
            regression_data = df[[target_column] + feature_columns].dropna()
            
            if len(regression_data) < len(feature_columns) + 1:
                return {
                    "type": "linear_regression",
                    "error": f"Not enough data points after removing missing values. Need at least {len(feature_columns) + 1} points, got {len(regression_data)}.",
                    "data": {}
                }
        
            # 分离目标变量
            y = regression_data[target_column]
            
            # 检查目标变量是否为数值类型
            if not np.issubdtype(y.dtype, np.number):
                return {
                    "type": "linear_regression",
                    "error": f"Target column '{target_column}' must be numeric.",
                    "data": {}
                }
            
            # 处理特征变量：编码分类变量和处理数值变量
            X_processed, encoding_info = self._prepare_features_with_encoding(
                regression_data, feature_columns, categorical_features, numerical_features, interaction_analysis
            )
            
            logger.info(f"处理后的特征数量: {X_processed.shape[1]}")
            logger.info(f"编码信息: {encoding_info}")
            
            # 执行线性回归
            analysis_result = self._perform_linear_regression(X_processed, y, target_column, feature_columns, intent, encoding_info, regression_data)
            
            return {
                "type": "linear_regression",
                "target_column": target_column,
                "feature_columns": feature_columns,
                "categorical_features": categorical_features,
                "numerical_features": numerical_features,
                "encoding_info": encoding_info,
                "data": analysis_result
            }
            
        except Exception as e:
            logger.error(f"Error in linear regression analysis: {e}")
            logger.error(traceback.format_exc())
            return {
                "type": "linear_regression",
                "error": f"Error in linear regression analysis: {str(e)}",
                "data": {}
            }

    def _identify_variable_types(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[List[str], List[str]]:
        """自动识别分类变量和数值变量"""
        categorical_features = []
        numerical_features = []
        
        for col in feature_columns:
            if col not in df.columns:
                continue
                
            col_data = df[col]
            
            # 判断是否为分类变量
            if (col_data.dtype == 'object' or 
                pd.api.types.is_categorical_dtype(col_data) or
                (col_data.nunique() < 20 and col_data.dtype != 'float64')):
                categorical_features.append(col)
            elif np.issubdtype(col_data.dtype, np.number):
                numerical_features.append(col)
            else:
                # 默认当作分类变量处理
                categorical_features.append(col)
        
        return categorical_features, numerical_features
    
    def _prepare_features_with_encoding(self, df: pd.DataFrame, feature_columns: List[str], 
                                       categorical_features: List[str], numerical_features: List[str],
                                       interaction_analysis: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """准备特征变量，包括分类变量编码和交互项生成"""
        
        encoding_info = {
            "categorical_encoding": {},
            "interaction_terms": [],
            "final_features": []
        }
        
        # 开始构建特征矩阵
        feature_dfs = []
        
        # 1. 处理数值变量
        if numerical_features:
            numerical_df = df[numerical_features].copy()
            feature_dfs.append(numerical_df)
            encoding_info["final_features"].extend(numerical_features)
            logger.info(f"添加数值变量: {numerical_features}")
        
        # 2. 处理分类变量（One-hot编码）
        for cat_col in categorical_features:
            if cat_col not in df.columns:
                continue
                
            # 执行One-hot编码，drop_first=True避免虚拟变量陷阱
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)
            feature_dfs.append(dummies)
            
            # 记录编码信息
            # 获取所有类别并排序，因为pd.get_dummies(drop_first=True)会删除排序后的第一个
            # 注意：pandas的get_dummies默认按字典序排序
            all_categories = sorted(df[cat_col].unique())
            
            # 验证实际的参考类别：通过检查哪个类别没有对应的虚拟变量
            dummy_categories = [col.replace(f"{cat_col}_", "") for col in dummies.columns]
            reference_category = None
            for cat in all_categories:
                if cat not in dummy_categories:
                    reference_category = cat
                    break
            
            # 如果没找到，使用排序后的第一个作为备选
            if reference_category is None:
                reference_category = all_categories[0]
            
            encoding_info["categorical_encoding"][cat_col] = {
                "method": "one_hot",
                "original_categories": df[cat_col].unique().tolist(),
                "dummy_columns": dummies.columns.tolist(),
                "reference_category": reference_category
            }
            encoding_info["final_features"].extend(dummies.columns.tolist())
            logger.info(f"One-hot编码 {cat_col}: {dummies.columns.tolist()}")
        
        # 3. 生成交互项
        include_interactions = interaction_analysis.get("include_interactions", True)
        if include_interactions and categorical_features and numerical_features:
            interaction_terms = self._generate_interaction_terms(
                df, categorical_features, numerical_features, encoding_info["categorical_encoding"]
            )
            
            if interaction_terms is not None and not interaction_terms.empty:
                feature_dfs.append(interaction_terms)
                encoding_info["interaction_terms"] = interaction_terms.columns.tolist()
                encoding_info["final_features"].extend(interaction_terms.columns.tolist())
                logger.info(f"添加交互项: {interaction_terms.columns.tolist()}")
        
        # 4. 合并所有特征
        if feature_dfs:
            X_processed = pd.concat(feature_dfs, axis=1)
            # 确保所有特征都是数值类型
            X_processed = X_processed.astype(float)
        else:
            return pd.DataFrame(), encoding_info
        
        logger.info(f"最终特征列: {X_processed.columns.tolist()}")
        return X_processed, encoding_info
    
    def _generate_interaction_terms(self, df: pd.DataFrame, categorical_features: List[str], 
                                   numerical_features: List[str], categorical_encoding: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """生成分类变量与数值变量之间的交互项"""
        interaction_dfs = []
        
        for cat_col in categorical_features:
            if cat_col not in categorical_encoding:
                continue
                
            dummy_columns = categorical_encoding[cat_col]["dummy_columns"]
            
            for num_col in numerical_features:
                if num_col not in df.columns:
                    continue
                    
                # 为每个虚拟变量与数值变量创建交互项
                for dummy_col in dummy_columns:
                    # 重新生成虚拟变量
                    dummy_data = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)[dummy_col]
                    
                    # 创建交互项
                    interaction_col = f"{dummy_col}_x_{num_col}"
                    interaction_data = dummy_data * df[num_col]
                    
                    interaction_df = pd.DataFrame({interaction_col: interaction_data})
                    interaction_dfs.append(interaction_df)
        
        if interaction_dfs:
            return pd.concat(interaction_dfs, axis=1)
        else:
            return None

    def _auto_identify_regression_variables(self, df: pd.DataFrame, data_understanding_insights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """自动识别回归分析的目标变量和特征变量"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            return {"target_column": None, "feature_columns": []}
        
        # 如果有数据理解洞察，使用它来更好地选择变量
        if data_understanding_insights and "column_analysis" in data_understanding_insights:
            # 根据LLM的语义理解选择变量
            potential_targets = []
            potential_features = []
            
            for col, details in data_understanding_insights["column_analysis"].items():
                if col not in numeric_columns:
                    continue
                    
                semantic_type = details.get("llm_semantic_type", "")
                
                # 根据语义类型分类
                if any(keyword in semantic_type.lower() for keyword in ["target", "dependent", "outcome", "result", "performance"]):
                    potential_targets.append(col)
                elif any(keyword in semantic_type.lower() for keyword in ["feature", "independent", "predictor", "input", "factor"]):
                    potential_features.append(col)
            
            if potential_targets and potential_features:
                return {
                    "target_column": potential_targets[0],
                    "feature_columns": potential_features[:3]  # 限制特征数量
                }
        
        # 默认策略：选择第一个数值列作为目标，其余作为特征
        return {
            "target_column": numeric_columns[0],
            "feature_columns": numeric_columns[1:4]  # 最多3个特征
        }
    
    def _perform_linear_regression(self, X: pd.DataFrame, y: pd.Series, target_column: str, feature_columns: List[str], intent: Dict[str, Any] = None, encoding_info: Dict[str, Any] = None, original_df: pd.DataFrame = None) -> Dict[str, Any]:
        """执行线性回归分析（用全部数据做回归）并评估变量适用性"""
        
        # 确保X是数值类型
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # 强制转换为数值类型
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # 检查是否有NaN值
        if X.isnull().any().any():
            logger.warning("发现NaN值，将被删除")
            # 找到有效行
            valid_rows = ~X.isnull().any(axis=1)
            X = X[valid_rows]
            y = y[valid_rows]
        
        # 转换为numpy数组以确保数值计算
        X_values = X.values.astype(float)
        y_values = y.values.astype(float)
        
        # 用全部数据训练模型
        model = LinearRegression()
        model.fit(X_values, y_values)
        y_pred = model.predict(X_values)
        
        # 计算全量数据的性能指标
        r2 = r2_score(y_values, y_pred)
        mse = mean_squared_error(y_values, y_pred)
        mae = mean_absolute_error(y_values, y_pred)
        rmse = np.sqrt(mse)
        
        # 获取系数
        coefficients = model.coef_
        intercept = model.intercept_
        
        # 获取实际的特征名称（编码后的）
        actual_feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        
        # 计算统计显著性（p值）
        from scipy import stats
        n = len(X_values)
        p = len(actual_feature_names)  # 使用实际特征数量
        
        # 计算残差
        residuals = y_values - y_pred
        
        # 计算均方误差
        mse_residual = np.sum(residuals**2) / (n - p - 1)
        
        # 计算X的转置点乘X的逆矩阵
        X_with_intercept = np.column_stack([np.ones(n), X_values])
        try:
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
        
        # 计算标准误差
        se_coefficients = np.sqrt(np.diagonal(XtX_inv) * mse_residual)
        
        # 计算t统计量和p值
        t_stats = np.array([intercept] + list(coefficients)) / se_coefficients
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        
        # 创建特征重要性字典（包含统计显著性）
        feature_importance = {}
        for i, feature in enumerate(actual_feature_names):
            p_value = float(p_values[i + 1])  # +1 因为第一个是截距
            t_stat = float(t_stats[i + 1])
            se = float(se_coefficients[i + 1])
            
            # 找到原始特征名称用于评估
            original_feature = self._get_original_feature_name(feature, encoding_info)
            
            # 评估变量适用性
            relevance = self._assess_feature_relevance(
                original_feature, p_value, coefficients[i], intent
            )
            
            feature_importance[feature] = {
                "coefficient": float(coefficients[i]),
                "abs_coefficient": float(abs(coefficients[i])),
                "scaled_importance": float(abs(coefficients[i]) / sum(abs(coefficients))) if sum(abs(coefficients)) != 0 else 0.0,
                "p_value": p_value,
                "t_statistic": t_stat,
                "standard_error": se,
                "is_significant": p_value < 0.05,
                "relevance": relevance["level"],
                "relevance_reason": relevance["reason"],
                "suggestion": relevance["suggestion"],
                "original_feature": original_feature,
                "feature_type": self._get_feature_type(feature, encoding_info)
            }
        
        # 预测结果
        predictions = []
        for i, (idx, actual) in enumerate(y.items()):
            predictions.append({
                "index": int(idx),
                "actual": round(float(actual), 3),
                "predicted": round(float(y_pred[i]), 3),
                "residual": round(float(actual - y_pred[i]), 3)
            })
        
        # 残差分析
        residual_stats = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max())
        }
        
        # 准备原始数据用于可视化（需要映射回原始特征）
        raw_data = []
        if original_df is not None:
            for i, (idx, actual) in enumerate(y.items()):
                data_point = {
                    "index": int(idx),
                    "actual": round(float(actual), 3),
                    "predicted": round(float(y_pred[i]), 3),
                    "residual": round(float(actual - y_pred[i]), 3)
                }
                # 添加原始特征变量的值（用于可视化）
                for original_feature in feature_columns:
                    if original_feature in original_df.columns:
                        # 从原始数据中获取对应索引的值
                        original_value = original_df.loc[idx, original_feature]
                        # 如果是数值类型，直接使用；如果是分类变量，转换为字符串
                        if pd.api.types.is_numeric_dtype(original_df[original_feature]):
                            data_point[original_feature] = round(float(original_value), 3)
                        else:
                            data_point[original_feature] = str(original_value)
                raw_data.append(data_point)
        
        # 变量适用性总结
        variable_assessment = self._summarize_variable_assessment(feature_importance)
        
        return {
            "model_info": {
                "type": "Linear Regression with Categorical Variables",
                "target_column": target_column,
                "feature_columns": feature_columns,  # 原始特征列
                "actual_features": actual_feature_names,  # 编码后的特征列
                "intercept": float(intercept),
                "feature_importance": feature_importance,
                "encoding_info": encoding_info
            },
            "performance_metrics": {
                "r2": float(r2),
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse)
            },
            "predictions": predictions,
            "raw_data": raw_data,  # 添加原始数据
            "residual_analysis": residual_stats,
            "data_summary": {
                "total_samples": len(X_values),
                "features_count": len(actual_feature_names),
                "original_features_count": len(feature_columns)
            },
            "variable_assessment": variable_assessment
        }
    
    def _get_original_feature_name(self, encoded_feature: str, encoding_info: Dict[str, Any]) -> str:
        """根据编码后的特征名称获取原始特征名称"""
        if not encoding_info:
            return encoded_feature
        
        # 检查是否是交互项
        if "_x_" in encoded_feature:
            return encoded_feature  # 交互项保持原样
        
        # 检查是否是分类变量的虚拟变量
        categorical_encoding = encoding_info.get("categorical_encoding", {})
        for original_cat, cat_info in categorical_encoding.items():
            if encoded_feature in cat_info.get("dummy_columns", []):
                return original_cat
        
        # 如果不是编码变量，返回原名
        return encoded_feature
    
    def _get_feature_type(self, encoded_feature: str, encoding_info: Dict[str, Any]) -> str:
        """获取特征类型"""
        if not encoding_info:
            return "numerical"
        
        # 检查是否是交互项
        if "_x_" in encoded_feature:
            return "interaction"
        
        # 检查是否是分类变量的虚拟变量
        categorical_encoding = encoding_info.get("categorical_encoding", {})
        for original_cat, cat_info in categorical_encoding.items():
            if encoded_feature in cat_info.get("dummy_columns", []):
                return "categorical_dummy"
        
        # 默认为数值型
        return "numerical"
    
    def _assess_feature_relevance(self, feature: str, p_value: float, coefficient: float, intent: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估特征变量的适用性"""
        # 获取意图中的变量评估信息
        intent_assessment = {}
        if intent and "feature_assessment" in intent:
            intent_assessment = intent.get("feature_assessment", {}).get(feature, {})
        
        # 基于统计显著性评估
        if p_value > 0.1:
            level = "low"
            reason = f"统计上不显著 (p值 = {p_value:.4f} > 0.1)"
            suggestion = "考虑从模型中移除该变量"
        elif p_value > 0.05:
            level = "medium"
            reason = f"边际显著 (p值 = {p_value:.4f})"
            suggestion = "谨慎使用，可能需要更多数据"
        else:
            level = "high"
            reason = f"统计上显著 (p值 = {p_value:.4f} < 0.05)"
            suggestion = "变量对模型有显著贡献"
        
        # 如果意图中有预先评估，结合考虑
        if intent_assessment:
            intent_level = intent_assessment.get("relevance", "")
            intent_reason = intent_assessment.get("reason", "")
            
            # 如果意图评估认为变量无关，但统计上显著，给出警告
            if intent_level == "irrelevant" and p_value < 0.05:
                level = "medium"
                reason = f"意图评估认为无关，但统计上显著 (p值 = {p_value:.4f})"
                suggestion = "需要进一步验证变量与目标变量的理论关系"
            elif intent_level == "irrelevant":
                level = "irrelevant"
                reason = f"意图评估认为无关: {intent_reason}"
                suggestion = intent_assessment.get("suggestion", "建议从模型中移除")

        return {
            "level": level,
            "reason": reason,
            "suggestion": suggestion
        }
    
    def _summarize_variable_assessment(self, feature_importance: Dict[str, Any]) -> Dict[str, Any]:
        """总结变量适用性评估"""
        irrelevant_vars = []
        low_relevance_vars = []
        medium_relevance_vars = []
        high_relevance_vars = []
        
        for feature, info in feature_importance.items():
            relevance = info.get("relevance", "unknown")
            if relevance == "irrelevant":
                irrelevant_vars.append(feature)
            elif relevance == "low":
                low_relevance_vars.append(feature)
            elif relevance == "medium":
                medium_relevance_vars.append(feature)
            elif relevance == "high":
                high_relevance_vars.append(feature)

        return {
            "irrelevant_variables": irrelevant_vars,
            "low_relevance_variables": low_relevance_vars,
            "medium_relevance_variables": medium_relevance_vars,
            "high_relevance_variables": high_relevance_vars,
            "summary": {
                "total_features": len(feature_importance),
                "high_relevance_count": len(high_relevance_vars),
                "medium_relevance_count": len(medium_relevance_vars),
                "low_relevance_count": len(low_relevance_vars),
                "irrelevant_count": len(irrelevant_vars)
            }
        }

