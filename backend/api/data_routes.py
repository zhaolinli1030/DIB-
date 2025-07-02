from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
from typing import Optional, Dict, Any
import logging
import traceback
import numpy as np

from ..orchestrator.orchestrator import get_orchestrator
from ..config import settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """将NumPy类型转换为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # 处理无穷大和NaN值
        if np.isinf(obj) or np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

router = APIRouter()

@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    上传数据文件并进行分析
    """
    try:
        logger.info(f"开始处理文件上传: {file.filename}，Content-Type: {file.content_type}")
        
        # 验证文件类型
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            logger.error(f"不支持的文件类型: {file.filename}")
            raise HTTPException(status_code=400, detail="只支持CSV和Excel文件")
        
        # 读取文件内容
        try:
            content = await file.read()
            logger.info(f"文件读取完成，内容大小: {len(content)} 字节")
            
            if len(content) > settings.MAX_UPLOAD_SIZE:
                logger.error(f"文件过大: {file.filename}")
                raise HTTPException(status_code=400, detail="文件大小不能超过10MB")
            
            logger.info(f"文件大小: {len(content)/1024/1024:.2f}MB")
            
        except Exception as e:
            logger.error(f"文件读取错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"文件读取错误: {str(e)}")
        
        # 读取文件内容
        try:
            logger.info(f"准备解析文件: {file.filename}")
            try:
                if file.filename.endswith('.csv'):
                    logger.info("使用pandas读取CSV文件")
                    df = pd.read_csv(pd.io.common.BytesIO(content))
                else:
                    logger.info("使用pandas读取Excel文件")
                    df = pd.read_excel(pd.io.common.BytesIO(content))
                
                logger.info(f"成功读取文件，数据形状: {df.shape}")
                logger.info(f"列名: {df.columns.tolist()}")
                
            except pd.errors.EmptyDataError:
                logger.error("文件为空")
                raise HTTPException(status_code=400, detail="文件为空")
            except pd.errors.ParserError as e:
                logger.error(f"文件解析错误: {str(e)}")
                raise HTTPException(status_code=400, detail=f"文件格式错误: {str(e)}")
            except Exception as e:
                logger.error(f"文件读取错误: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=400, detail=f"文件读取错误: {str(e)}")
            
        except Exception as e:
            logger.error(f"文件解析错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"文件解析错误: {str(e)}")
        
        # 如果没有提供session_id，创建一个新的会话
        try:
            if not session_id:
                session_id = await orchestrator.create_session()
                logger.info(f"创建新会话: {session_id}")
        except Exception as e:
            logger.error(f"会话创建错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"会话创建错误: {str(e)}")
        
        # 处理数据
        try:
            logger.info("开始处理数据...")
            result = await orchestrator.upload_data(session_id, df)
            logger.info("数据处理完成")
            
            # 转换NumPy类型为Python原生类型
            result = convert_numpy_types(result)
            
            # 返回结果
            response_data = {
                "session_id": session_id,
                "filename": file.filename,
                "rows": int(len(df)),  # 确保是Python int类型
                "columns": int(len(df.columns)),  # 确保是Python int类型
                "analysis": result
            }
            logger.info(f"成功完成上传，返回数据: {response_data}")
            return response_data
            
        except Exception as e:
            logger.error(f"数据处理错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"数据处理错误: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"未预期的错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 添加OPTIONS路由处理
@router.options("/upload")
async def options_upload():
    return {}  # 提供空响应，让CORS中间件添加适当的头部

@router.get("/{session_id}/summary")
async def get_data_summary(
    session_id: str,
    orchestrator = Depends(get_orchestrator)
):
    """获取数据分析摘要"""
    session_data = orchestrator.get_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session_data.get("data_analysis"):
        # 返回空的分析摘要，而不是抛出错误
        return {
            "basic_info": {},
            "quality_analysis": {},
            "analysis_suggestions": [],
            "message": "No data analysis available. Please upload data first."
        }
    
    analysis = session_data["data_analysis"]
    # 转换NumPy类型为Python原生类型
    analysis = convert_numpy_types(analysis)
    
    return {
        "basic_info": analysis["basic_info"],
        "quality_analysis": analysis["quality_analysis"],
        "analysis_suggestions": analysis["analysis_suggestions"]
    }

def safe_float(value):
    """安全地转换浮点数值，处理特殊情况"""
    try:
        if pd.isna(value) or np.isinf(value) or np.isnan(value):
            return None
        # 检查是否超出JSON可表示的范围
        if abs(value) > 1e308:  # JSON最大可表示的数约为1e308
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

@router.get("/{session_id}/preview")
async def get_data_preview(
    session_id: str,
    rows: int = 10,
    orchestrator = Depends(get_orchestrator)
):
    """获取数据预览"""
    try:
        logger.info(f"获取数据预览，session_id: {session_id}, rows: {rows}")
        
        session_data = orchestrator.get_session_data(session_id)
        if not session_data:
            logger.error(f"会话不存在: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
            
        if session_data.get("data") is None:
            logger.warning(f"会话中没有数据: {session_id}")
            # 返回空的数据预览，而不是抛出错误
            return {
                "columns": [],
                "data": [],
                "total_rows": 0,
                "message": "No data available in session. Please upload data first."
            }
        
        df = session_data["data"]
        logger.info(f"获取到数据，形状: {df.shape}")
        
        try:
            # 将DataFrame转换为字典，并处理数值
            preview = df.head(rows).to_dict(orient="records")
            logger.info(f"成功生成预览数据，记录数: {len(preview)}")
            
            # 转换NumPy类型为Python原生类型，并处理特殊值
            processed_preview = []
            for record in preview:
                processed_record = {}
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.floating)):
                        processed_record[key] = safe_float(value)
                    elif isinstance(value, (int, float)):
                        processed_record[key] = safe_float(value)
                    else:
                        processed_record[key] = value
                processed_preview.append(processed_record)
            
            response_data = {
                "columns": df.columns.tolist(),
                "data": processed_preview,
                "total_rows": int(len(df))
            }
            logger.info("成功返回预览数据")
            return response_data
            
        except Exception as e:
            logger.error(f"生成预览数据时出错: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据预览时出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/{session_id}/columns")
async def get_column_details(
    session_id: str,
    orchestrator = Depends(get_orchestrator)
):
    """获取列详细信息"""
    session_data = orchestrator.get_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session_data.get("data_analysis"):
        # 返回空的列详情，而不是抛出错误
        return {
            "message": "No data analysis available. Please upload data first."
        }
    
    column_analysis = session_data["data_analysis"]["column_analysis"]
    # 转换NumPy类型为Python原生类型
    column_analysis = convert_numpy_types(column_analysis)
    
    return column_analysis