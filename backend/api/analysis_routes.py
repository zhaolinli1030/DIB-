from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import traceback
import json
import uuid

from ..orchestrator.orchestrator import get_orchestrator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    """分析查询请求模型"""
    query: str

@router.post("/{session_id}/query")
async def submit_analysis_query(
    session_id: str,
    request: QueryRequest,
    orchestrator = Depends(get_orchestrator)
):
    """提交自然语言分析查询"""
    try:
        query = request.query
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        logger.info(f"Processing analysis query for session {session_id}: {query}")
        
        # 使用orchestrator处理查询
        result = await orchestrator.process_query(session_id, query)
        
        if not result:
            raise HTTPException(status_code=500, detail="No result returned from query processing")
        
        # 生成analysis_id (如果结果中没有的话)
        analysis_id = result.get("analysis_id")
        if not analysis_id:
            analysis_id = str(uuid.uuid4())
            result["analysis_id"] = analysis_id
        
        return {
            "success": True, 
            "analysis_id": analysis_id,
            "result": result
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/{analysis_id}/result")
async def get_analysis_result(
    analysis_id: str,
    session_id: str,
    orchestrator = Depends(get_orchestrator)
):
    """获取分析结果详情"""
    try:
        logger.info(f"Getting analysis result for analysis_id: {analysis_id}, session_id: {session_id}")
        
        analysis = orchestrator.get_session_analysis(session_id)
        if not analysis:
            logger.error(f"No analysis found for session: {session_id}")
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        if analysis.get("analysis_id") != analysis_id:
            logger.error(f"Analysis ID mismatch. Expected: {analysis_id}, Got: {analysis.get('analysis_id')}")
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        return {
            "analysis_id": analysis_id,
            "analysis_result": analysis["analysis"],
            "visualization": analysis["visualization"],
            "interpretation": analysis["interpretation"]
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting analysis result: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting analysis result: {str(e)}")

@router.get("/{session_id}/suggestions")
async def get_analysis_suggestions(
    session_id: str,
    orchestrator = Depends(get_orchestrator)
):
    """获取分析建议"""
    try:
        logger.info(f"Getting analysis suggestions for session: {session_id}")
        
        session_data = orchestrator.get_session_data(session_id)
        if not session_data:
            logger.error(f"Session not found for suggestions: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 尝试从session_data中获取分析建议
        suggestions = []
        if "data_analysis" in session_data and "analysis_suggestions" in session_data["data_analysis"]:
            suggestions = session_data["data_analysis"]["analysis_suggestions"]
        else:
            # 如果没有预生成的建议，创建默认建议
            logger.info("No pre-generated suggestions found, creating default ones")
            suggestions = [
                {
                    "type": "基本统计分析",
                    "description": "查看数据的基本统计信息和分布",
                    "action": "basic_stats"
                },
                {
                    "type": "数据质量检查", 
                    "description": "检查缺失值、异常值和数据完整性",
                    "action": "data_quality"
                },
                {
                    "type": "相关性分析",
                    "description": "分析变量之间的关联关系",
                    "action": "correlation"
                }
            ]
        
        logger.info(f"Returning {len(suggestions)} suggestions")
        return {"suggestions": suggestions}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis suggestions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"获取分析建议失败: {str(e)}")

@router.get("/{session_id}/status")
async def get_session_status(
    session_id: str,
    orchestrator = Depends(get_orchestrator)
):
    """获取会话状态（调试用）"""
    try:
        session_data = orchestrator.get_session_data(session_id)
        
        if session_data:
            return {
                "session_id": session_id,
                "exists": True,
                "has_data": "data" in session_data,
                "has_analysis": "data_analysis" in session_data,
                "data_shape": session_data["data"].shape if "data" in session_data else None,
                "created_at": session_data.get("created_at"),
                "updated_at": session_data.get("updated_at")
            }
        else:
            return {
                "session_id": session_id,
                "exists": False
            }
    except Exception as e:
        logger.error(f"Error checking session status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/history")
async def get_analysis_history(
    session_id: str,
    orchestrator = Depends(get_orchestrator)
):
    """获取分析历史"""
    try:
        session_data = orchestrator.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        query_history = session_data.get("query_history", [])
        return {
            "history": query_history
        }
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))