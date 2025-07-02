import pandas as pd
import io
import os # os is imported but not used, consider removing if not needed elsewhere.
from typing import Dict, Any, Optional, Tuple
from fastapi import UploadFile
import json

# 导入 DataProcessor 以使用其 _clean_column_names 方法
from ..data.processor import DataProcessor

class DataLoader:
    """负责加载和初步处理数据文件 (Responsible for loading and initial processing of data files)"""
    
    @staticmethod
    async def load_from_upload(file: UploadFile) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        从上传的文件加载数据 (Load data from an uploaded file)
        
        Args:
            file: 上传的文件对象 (Uploaded file object)
            
        Returns:
            加载的DataFrame和元数据 (Loaded DataFrame and metadata)
        """
        # 读取文件内容 (Read file content)
        content = await file.read()
        
        # 基于文件类型处理数据 (Process data based on file type)
        file_extension = file.filename.split(".")[-1].lower()
        
        df = None
        metadata_extra = {} # To store file-type specific metadata

        if file_extension == "csv":
            df, metadata_extra = DataLoader._process_csv(content)
        elif file_extension in ["xlsx", "xls"]:
            df, metadata_extra = DataLoader._process_excel(content)
        elif file_extension == "json":
            df, metadata_extra = DataLoader._process_json(content)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # 清理列名 (Clean column names)
        if df is not None:
            df = DataProcessor._clean_column_names(df) # <--- 调用 DataProcessor 中的方法

        # 创建完整的元数据 (Create complete metadata)
        metadata = {
            "file_type": file_extension,
            "rows": len(df) if df is not None else 0,
            "columns": len(df.columns) if df is not None and hasattr(df, 'columns') else 0,
            **metadata_extra # 合并特定文件类型的元数据 (Merge file-type specific metadata)
        }
        
        return df, metadata
    
    @staticmethod
    def _process_csv(content: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """处理CSV文件内容 (Process CSV file content)"""
        # 尝试使用不同的编码和分隔符读取CSV
        # Try to read CSV using different encodings and separators
        encodings = ["utf-8", "latin1", "cp1252", "gbk"] # Added gbk for broader Chinese compatibility
        separators = [",", ";", "\t", "|"]
        
        df = None
        used_encoding = "utf-8" # Default
        used_separator = "," # Default
        
        # 尝试不同的编码和分隔符组合
        # Try different combinations of encodings and separators
        for encoding_attempt in encodings:
            if df is not None: # If a valid df is found, break
                break
            try:
                for sep_attempt in separators:
                    try:
                        # Use a BytesIO stream for pandas
                        current_df = pd.read_csv(
                            io.BytesIO(content), 
                            sep=sep_attempt, 
                            encoding=encoding_attempt,
                            engine="python", # Using python engine for flexibility with separators
                            skipinitialspace=True # Handles cases with spaces after delimiter
                        )
                        
                        # 如果读取成功且列数大于0 (或大于1，如果期望多列)
                        # 且至少有一行数据，或者即使没数据但有列
                        # If read successfully and has columns (or more than 1 if multiple are expected)
                        # and has at least one row, or has columns even if no data
                        if current_df.shape[1] > 0 or (current_df.shape[1] == 0 and not current_df.empty):
                            # 更鲁棒的检查：如果分隔符真的分割出了多列，或者只有一列但确实是预期的
                            if current_df.shape[1] > 1 or \
                               (current_df.shape[1] == 1 and len(current_df.columns) > 0 and sep_attempt not in str(current_df.columns[0])): # Check if separator is part of the single column name
                                df = current_df
                                used_encoding = encoding_attempt
                                used_separator = sep_attempt
                                break # Found a good combination for separator
                    except UnicodeDecodeError: # Specific error for encoding issues
                        continue # Try next encoding or separator
                    except pd.errors.EmptyDataError: # File is empty
                        df = pd.DataFrame() # Return an empty DataFrame
                        used_encoding = encoding_attempt # Keep the attempted encoding
                        break # Stop further attempts for empty file
                    except Exception: # Catch other parsing errors (e.g., CParserError)
                        continue # Try next separator or encoding
                if df is not None: # If df found in inner loop, break outer loop
                    break
            except Exception: 
                continue # Try next encoding
        
        if df is None:
            # 如果所有尝试都失败，使用默认设置并接受可能的结果
            try:
                df = pd.read_csv(io.BytesIO(content))
                # used_encoding 和 used_separator 保持默认 (used_encoding and used_separator remain default)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame() # Handle empty file case for default read
            except Exception as e:
                 raise ValueError(f"Failed to parse CSV with all attempted encodings/separators and default settings: {e}")

        # CSV特定的元数据 (CSV specific metadata)
        metadata_extra = {
            "encoding": used_encoding,
            "separator": used_separator
        }
        
        return df, metadata_extra
    
    @staticmethod
    def _process_excel(content: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """处理Excel文件内容 (Process Excel file content)"""
        try:
            # 读取Excel文件，默认使用第一个工作表
            # Read Excel file, defaults to the first sheet
            df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}")
        
        # Excel特定的元数据 (Excel specific metadata - none for now, but can be added)
        metadata_extra = {}
        
        return df, metadata_extra
    
    @staticmethod
    def _process_json(content: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """处理JSON文件内容 (Process JSON file content)"""
        df = None
        try:
            # 尝试将JSON字符串解码
            # Try to decode JSON string
            json_str = content.decode('utf-8')
            data = json.loads(json_str)

            # 尝试使用 pd.json_normalize，它对各种嵌套结构更鲁棒
            # Try using pd.json_normalize, which is more robust for various nested structures
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                # 如果是字典，尝试找到一个主列表进行规范化，否则规范化整个字典
                # If it's a dictionary, try to find a main list to normalize,
                # otherwise normalize the entire dictionary.
                # This part can be complex depending on expected JSON structures.
                # For simplicity, we'll normalize the whole dict if no obvious list of records.
                df = pd.json_normalize(data)
            else:
                raise ValueError("JSON content is not a list or dictionary of records.")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"JSON encoding error (expected UTF-8): {e}")
        except Exception as e: # Catch other potential errors during normalization
            # Fallback to pd.read_json if json_normalize fails or for simpler structures
            try:
                df = pd.read_json(io.BytesIO(content), orient='records', lines=False)
            except ValueError:
                try:
                    df = pd.read_json(io.BytesIO(content), orient='records', lines=True)
                except Exception as final_e:
                    raise ValueError(f"Failed to parse JSON with multiple attempts: Initial error ({e}), Final error ({final_e})")
        
        if df is None: # Should not happen if logic is correct, but as a safeguard
            raise ValueError("Could not parse JSON into DataFrame")

        # JSON特定的元数据 (JSON specific metadata)
        metadata_extra = {}
        
        return df, metadata_extra