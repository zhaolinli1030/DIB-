from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .api.data_routes import router as data_router
from .api.analysis_routes import router as analysis_router
from .api.session_routes import router as session_router # <--- 添加 session_router 导入
from .config import settings # <--- 添加 settings 导入

# 配置日志 (Configure logging)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DataInsightBot API")

# 配置CORS (Configure CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # <--- 修改: 使用配置文件中的设置
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法 
    allow_headers=["*"],  # 允许所有headers
)

# 注册路由 (Register routes)
app.include_router(data_router, prefix="/api/data", tags=["data"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])
app.include_router(session_router, prefix="/api/session", tags=["session"]) # <--- 添加 session_router 的注册

@app.on_event("startup")
async def startup_event():
    logger.info("Starting DataInsightBot API...")
    # 可以在这里初始化或检查 get_orchestrator() 来确保单例创建
    from .orchestrator.orchestrator import get_orchestrator
    get_orchestrator()


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down DataInsightBot API...")