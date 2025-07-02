from typing import Dict, Any, List, Optional
import uuid
import asyncio
from fastapi import HTTPException
import logging
import traceback
from datetime import datetime

from ..agents.data_agent import DataUnderstandingAgent
from ..agents.intent_agent import IntentUnderstandingAgent
from ..agents.analysis_agent import AnalysisExecutionAgent
from ..agents.viz_agent import VisualizationAgent
from ..agents.interpret_agent import InterpretationAgent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建全局orchestrator实例
_orchestrator_instance = None

def get_orchestrator():
    """获取全局orchestrator实例"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
        logger.info("Created new orchestrator instance")
    return _orchestrator_instance

class Orchestrator:
    """
    智能体协调器，负责管理智能体之间的协作和工作流控制
    """
    
    def __init__(self):
        # 初始化智能体
        self.data_agent = DataUnderstandingAgent()
        self.intent_agent = IntentUnderstandingAgent()
        self.analysis_agent = AnalysisExecutionAgent()
        self.viz_agent = VisualizationAgent()
        self.interpret_agent = InterpretationAgent()
        
        # 会话状态存储
        self.sessions = {}
        logger.info("Orchestrator initialized")
    
    async def create_session(self, session_id: Optional[str] = None) -> str:
        """创建新的会话并返回会话ID"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # 检查会话是否已存在
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, updating...")
            
            self.sessions[session_id] = {
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "last_active": datetime.now(),
                "data": None,  # 将存储处理后的DataFrame
                "data_analysis": None,  # 将存储数据分析结果
                "query_history": [],  # 查询历史
                "current_analysis": None,  # 当前分析任务
                "persistent": True  # 标记为持久会话，不会被自动清理
            }
            
            logger.info(f"Created/Updated session: {session_id}")
            logger.info(f"Current sessions count: {len(self.sessions)}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")
    
    async def upload_data(self, session_id: str, dataframe: Any) -> Dict[str, Any]:
        """
        处理上传的数据并进行初步分析
        
        Args:
            session_id: 会话ID
            dataframe: 已加载的DataFrame
            
        Returns:
            数据分析结果
        """
        try:
            logger.info(f"Processing data upload for session: {session_id}")
            
            # 确保会话存在
            if session_id not in self.sessions:
                logger.info(f"Session {session_id} not found, creating new session")
                await self.create_session(session_id)
            
            # 更新会话状态
            self.sessions[session_id]["last_active"] = datetime.now()
            self.sessions[session_id]["data"] = dataframe
            
            logger.info(f"DataFrame shape: {dataframe.shape}")
            
            # 使用数据理解智能体分析数据
            try:
                logger.info("Starting data analysis")
                data_analysis = await self.data_agent.process({"dataframe": dataframe})
                
                # 存储分析结果
                self.sessions[session_id]["data_analysis"] = data_analysis
                
                logger.info("Data analysis completed successfully")
                return data_analysis
                
            except Exception as e:
                logger.error(f"Data analysis error: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Data analysis error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Data processing error: {str(e)}")
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话数据"""
        try:
            logger.info(f"Getting session data for session: {session_id}")
            
            # 检查session_id是否存在
            if not session_id:
                logger.error("Empty session_id provided")
                return None
                
            # 检查sessions字典
            if not hasattr(self, 'sessions'):
                logger.error("Sessions storage not initialized")
                self.sessions = {}
                return None
                
            # 详细日志记录
            logger.info(f"Current sessions keys: {list(self.sessions.keys())}")
            
            if session_id in self.sessions:
                session_data = self.sessions[session_id]
                # 更新最后活动时间
                session_data["last_active"] = datetime.now()
                session_data["updated_at"] = datetime.now()
                
                # 检查是否有数据，但即使没有数据也返回会话信息（允许重新上传）
                if session_data.get("data") is None:
                    logger.warning(f"No data available for session: {session_id}, but session exists")
                    # 返回会话信息，允许重新上传数据
                    return {
                        "session_id": session_id,
                        "created_at": session_data.get("created_at"),
                        "last_active": session_data.get("last_active"),
                        "data": None,
                        "data_analysis": None,
                        "query_history": session_data.get("query_history", []),
                        "current_analysis": session_data.get("current_analysis"),
                        "persistent": session_data.get("persistent", True)
                    }
                    
                logger.info(f"Successfully retrieved session data for: {session_id}")
                logger.info(f"Session data keys: {list(session_data.keys()) if isinstance(session_data, dict) else 'Not a dict'}")
                return session_data
            else:
                logger.error(f"Session not found: {session_id}")
                logger.error(f"Available sessions: {list(self.sessions.keys())}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting session data: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def get_session_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话的当前分析结果"""
        try:
            logger.info(f"Getting session analysis for session: {session_id}")
            
            session = self.sessions.get(session_id)
            if not session:
                logger.error(f"Session not found: {session_id}")
                # 尝试重新创建会话
                try:
                    asyncio.create_task(self.create_session(session_id))
                    logger.info(f"Attempting to recreate session: {session_id}")
                except Exception as e:
                    logger.error(f"Failed to recreate session: {str(e)}")
                return None
                
            # 更新最后活动时间
            session["last_active"] = datetime.now()
            
            current_analysis = session.get("current_analysis")
            if not current_analysis:
                logger.warning(f"No current analysis available for session: {session_id}")
                # 如果有数据但没有分析结果，尝试重新分析
                if session.get("data") is not None and session.get("data_analysis") is None:
                    try:
                        asyncio.create_task(self.data_agent.process({"dataframe": session["data"]}))
                        logger.info(f"Attempting to reanalyze data for session: {session_id}")
                    except Exception as e:
                        logger.error(f"Failed to reanalyze data: {str(e)}")
                
            return current_analysis
            
        except Exception as e:
            logger.error(f"Error getting session analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def cleanup_sessions(self, max_age: float = 3600.0) -> int:
        """清理不活跃的会话 - 已禁用自动清理，会话将保持到服务器重启"""
        try:
            # 禁用自动会话清理，确保数据持久性
            logger.info("Session cleanup disabled - sessions will persist until server restart")
            return 0
            
        except Exception as e:
            logger.error(f"Error in cleanup_sessions: {str(e)}")
            logger.error(traceback.format_exc())
            return 0
    
    async def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            session_id: 会话ID
            query: 用户查询文本
            
        Returns:
            分析结果
        """
        try:
            logger.info(f"Processing query for session: {session_id}")
            logger.info(f"Query: {query}")
            
            # 检查sessions字典
            if not hasattr(self, 'sessions'):
                logger.error("Sessions storage not initialized")
                self.sessions = {}
                raise HTTPException(status_code=500, detail="Internal server error: sessions not initialized")
            
            # 详细日志记录当前会话状态
            logger.info(f"Current sessions keys: {list(self.sessions.keys())}")
            
            # 确保会话存在
            session = self.sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                logger.error(f"Available sessions: {list(self.sessions.keys())}")
                # 尝试重新创建会话
                try:
                    await self.create_session(session_id)
                    session = self.sessions[session_id]
                    logger.info(f"Successfully recreated session: {session_id}")
                except Exception as e:
                    logger.error(f"Failed to recreate session: {str(e)}")
                    raise HTTPException(status_code=404, detail="Session not found and could not be recreated")
            
            # 更新会话状态
            session["last_active"] = datetime.now()
            session["updated_at"] = datetime.now()
            
            # 检查是否有数据
            if session.get("data") is None:
                logger.error("No data available for analysis")
                raise HTTPException(status_code=400, detail="No data available for analysis")
            
            # 检查是否有数据分析结果
            if session.get("data_analysis") is None:
                logger.warning("No data analysis available, attempting to analyze data")
                try:
                    data_analysis = await self.data_agent.process({"dataframe": session["data"]})
                    session["data_analysis"] = data_analysis
                    logger.info("Successfully analyzed data")
                except Exception as e:
                    logger.error(f"Failed to analyze data: {str(e)}")
                    raise HTTPException(status_code=500, detail="Failed to analyze data")
            
            # 使用 asyncio.wait_for 添加超时控制
            try:
                # 1. 使用意图理解智能体解析查询
                logger.info("Starting intent understanding")
                intent_result = await asyncio.wait_for(
                    self.intent_agent.process({
                        "query": query,
                        "data_analysis": session["data_analysis"]
                    }),
                    timeout=600.0  # 增加到600秒超时
                )
                logger.info(f"Intent understanding completed: {intent_result}")
                
                # 2. 使用分析执行智能体执行分析
                logger.info("Starting analysis execution")
                analysis_result = await asyncio.wait_for(
                    self.analysis_agent.process({
                        "intent": intent_result,
                        "data": session["data"],
                        "data_analysis": session["data_analysis"]
                    }),
                    timeout=600.0  # 增加到600秒超时
                )
                logger.info("Analysis execution completed")
                
                # 3. 使用可视化智能体生成可视化
                logger.info("Starting visualization generation")
                viz_result = await asyncio.wait_for(
                    self.viz_agent.process({
                        "analysis_result": analysis_result,
                        "data": session["data"]
                    }),
                    timeout=600.0  # 增加到600秒超时
                )
                logger.info("Visualization generation completed")
                
                # 4. 使用解释智能体生成解释
                logger.info("Starting interpretation generation")
                interpretation = await asyncio.wait_for(
                    self.interpret_agent.process({
                        "analysis_result": analysis_result,
                        "visualization": viz_result,
                        "intent": intent_result,
                        "query": query
                    }),
                    timeout=600.0  # 增加到600秒超时
                )
                logger.info("Interpretation generation completed")
                
                # 生成分析ID
                analysis_id = str(uuid.uuid4())
                
                # 整合结果
                result = {
                    "analysis_id": analysis_id,
                    "intent": intent_result,
                    "analysis": analysis_result,
                    "visualization": viz_result,
                    "interpretation": interpretation
                }
                
                # 更新会话状态
                session["current_analysis"] = result
                session["query_history"].append({
                    "query": query,
                    "timestamp": datetime.now(),
                    "result": result
                })
                
                logger.info(f"Query processing completed successfully with analysis_id: {analysis_id}")
                return result
                
            except asyncio.TimeoutError:
                logger.error("Query processing timed out")
                raise HTTPException(status_code=504, detail="Query processing timed out")
            
        except HTTPException as he:
            logger.error(f"HTTP error processing query: {str(he)}")
            raise he
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")