import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class DataProcessor:
    """负责数据清洗和预处理"""
    
    @staticmethod
    def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        执行基本的数据预处理
        
        Args:
            df: 原始DataFrame
            
        Returns:
            预处理后的DataFrame和处理日志
        """
        processing_log = []
        processed_df = df.copy()
        
        # 1. 清理列名
        old_columns = processed_df.columns.tolist()
        processed_df = DataProcessor._clean_column_names(processed_df)
        new_columns = processed_df.columns.tolist()
        
        if old_columns != new_columns:
            processing_log.append({
                "step": "清理列名",
                "details": {"old_columns": old_columns, "new_columns": new_columns}
            })
        
        # 2. 检测并转换日期列
        date_conversions = DataProcessor._convert_date_columns(processed_df)
        if date_conversions:
            processing_log.append({
                "step": "日期列转换",
                "details": {"converted_columns": date_conversions}
            })
        
        # 3. 处理数值列中的异常值
        outlier_info = DataProcessor._handle_numeric_outliers(processed_df)
        if outlier_info:
            processing_log.append({
                "step": "异常值处理",
                "details": outlier_info
            })
        
        # 4. 处理缺失值
        missing_info = DataProcessor._handle_missing_values(processed_df)
        if missing_info:
            processing_log.append({
                "step": "缺失值处理",
                "details": missing_info
            })
        
        return processed_df, {"log": processing_log}
    
    @staticmethod
    def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """清理列名，使其更易于处理"""
        # 复制DataFrame以避免修改原始数据
        cleaned_df = df.copy()
        
        # 清理列名
        cleaned_df.columns = [
            str(col).strip()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace("{", "")
            .replace("}", "")
            .replace(":", "")
            .replace(";", "")
            .replace(",", "")
            .replace("'", "")
            .replace('"', "")
            .replace("/", "_")
            .replace("\\", "_")
            .lower()
            for col in cleaned_df.columns
        ]
        
        return cleaned_df
    
    @staticmethod
    def _convert_date_columns(df: pd.DataFrame) -> List[str]:
        """检测并转换可能的日期列"""
        converted_columns = []
        
        for column in df.columns:
            # 跳过已经是日期类型的列
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                continue
                
            # 只检查字符串列
            if df[column].dtype == 'object':
                # 尝试将前10个非空值转换为日期
                sample = df[column].dropna().head(10)
                
                try:
                    pd.to_datetime(sample, errors='raise')
                    # 如果转换成功，则转换整列
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    converted_columns.append(column)
                except:
                    pass
        
        return converted_columns
    
    @staticmethod
    def _handle_numeric_outliers(df: pd.DataFrame) -> Dict[str, Any]:
        """处理数值列中的异常值"""
        outlier_info = {"detected": {}, "handled": {}}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            col_data = df[column].dropna()
            
            # 如果数据点太少，则跳过
            if len(col_data) < 10:
                continue
            
            # 使用IQR方法检测异常值
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_info["detected"][column] = outlier_count
                
                # 对于MVP，我们只记录异常值但不处理它们
                # 在实际产品中，可以实现异常值替换或移除策略
                # df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
                # outlier_info["handled"][column] = outlier_count
        
        return outlier_info
    
    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
        """处理缺失值"""
        missing_info = {"before": {}, "after": {}}
        
        # 记录处理前的缺失值
        for column in df.columns:
            missing_count = df[column].isna().sum()
            if missing_count > 0:
                missing_info["before"][column] = int(missing_count)
        
        # 对于MVP，我们不实际处理缺失值，只记录它们
        # 在实际产品中，可以实现按列类型的缺失值填充策略
        
        # 记录处理后的缺失值（在MVP中与处理前相同）
        missing_info["after"] = missing_info["before"].copy()
        
        return missing_info
    
    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """分析并返回每列的数据类型"""
        column_types = {}
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                if df[column].dropna().astype(int).equals(df[column].dropna()):
                    column_types[column] = "integer"
                else:
                    column_types[column] = "float"
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                column_types[column] = "datetime"
            elif df[column].nunique() < 10 and len(df) > 20:
                column_types[column] = "categorical"
            else:
                column_types[column] = "text"
        
        return column_types