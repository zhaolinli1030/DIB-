import pandas as pd
from typing import Dict, Any, List, Tuple

class DataValidator:
    """负责验证数据文件和DataFrame的质量"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        验证DataFrame的基本质量
        
        Args:
            df: 要验证的DataFrame
            
        Returns:
            验证结果(True/False)和详细信息
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # 检查DataFrame是否为空
        if df.empty:
            validation_results["is_valid"] = False
            validation_results["errors"].append({
                "type": "empty_dataframe",
                "message": "DataFrame is empty"
            })
            return False, validation_results
        
        # 检查列名是否唯一
        if not df.columns.is_unique:
            validation_results["is_valid"] = False
            duplicated_columns = df.columns[df.columns.duplicated()].tolist()
            validation_results["errors"].append({
                "type": "duplicate_columns",
                "message": f"DataFrame contains duplicate column names: {duplicated_columns}"
            })
        
        # 检查是否存在全空的列
        empty_columns = [col for col in df.columns if df[col].isna().all()]
        if empty_columns:
            validation_results["warnings"].append({
                "type": "empty_columns",
                "message": f"DataFrame contains completely empty columns: {empty_columns}"
            })
        
        # 检查是否存在高比例缺失值的列
        high_missing_columns = []
        for col in df.columns:
            missing_ratio = df[col].isna().mean()
            if missing_ratio > 0.5:  # 超过50%的值缺失
                high_missing_columns.append({
                    "column": col,
                    "missing_ratio": missing_ratio
                })
        
        if high_missing_columns:
            validation_results["warnings"].append({
                "type": "high_missing_ratio",
                "message": "Some columns have high missing value ratios",
                "details": high_missing_columns
            })
        
        # 检查是否存在足够的数据行用于分析
        if len(df) < 5:
            validation_results["warnings"].append({
                "type": "few_rows",
                "message": f"DataFrame contains only {len(df)} rows, which may be insufficient for meaningful analysis"
            })
        
        # 检查是否有过多的列
        if len(df.columns) > 100:
            validation_results["warnings"].append({
                "type": "many_columns",
                "message": f"DataFrame contains {len(df.columns)} columns, which may make analysis difficult to interpret"
            })
        
        return validation_results["is_valid"], validation_results
    
    @staticmethod
    def get_column_quality(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        评估每列的数据质量
        
        Args:
            df: 要评估的DataFrame
            
        Returns:
            包含每列质量指标的字典
        """
        quality_metrics = {}
        
        for column in df.columns:
            column_data = df[column]
            metrics = {
                "missing_ratio": float(column_data.isna().mean()),
                "unique_ratio": float(column_data.nunique() / len(df)) if len(df) > 0 else 0
            }
            
            # 对数值列计算额外指标
            if pd.api.types.is_numeric_dtype(column_data):
                # 检查是否存在常量值
                non_na_values = column_data.dropna()
                if len(non_na_values) > 0 and non_na_values.std() == 0:
                    metrics["is_constant"] = True
                else:
                    metrics["is_constant"] = False
                
                # 检查异常值
                try:
                    q1 = column_data.quantile(0.25)
                    q3 = column_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outlier_ratio = ((column_data < lower_bound) | (column_data > upper_bound)).mean()
                    metrics["outlier_ratio"] = float(outlier_ratio)
                except:
                    metrics["outlier_ratio"] = None
            
            # 对分类列计算额外指标
            if column_data.dtype == 'object' or pd.api.types.is_categorical_dtype(column_data):
                # 检查是否是ID列
                if metrics["unique_ratio"] > 0.9:
                    metrics["potential_id"] = True
                else:
                    metrics["potential_id"] = False
                
                # 检查基数（唯一值的数量）
                metrics["cardinality"] = int(column_data.nunique())
            
            quality_metrics[column] = metrics
        
        return quality_metrics