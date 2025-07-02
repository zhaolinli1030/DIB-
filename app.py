import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.api import data_routes, analysis_routes, session_routes
from backend.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时的初始化代码
    print("🚀 Starting DataInsightBot - Linear Regression Analysis API...")
    print("📊 Focus: Linear Regression Analysis (Simple & Multiple)")
    print("🤖 AI Agents: Intent Recognition, Data Analysis, Model Training, Visualization")
    yield
    # 应用关闭时的清理代码
    print("🛑 Shutting down DataInsightBot API...")

app = FastAPI(
    title="DataInsightBot - Linear Regression Analysis API",
    description="智能线性回归分析助手，支持简单和多元线性回归，提供专业的模型性能评估和可视化",
    version="1.0.0",
    lifespan=lifespan
)

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 明确指定前端URL
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
    expose_headers=["*"]  # 暴露所有头
)

# 注册路由
app.include_router(session_routes.router, prefix="/api/session", tags=["Session"])
app.include_router(data_routes.router, prefix="/api/data", tags=["Data"])
app.include_router(analysis_routes.router, prefix="/api/analysis", tags=["Linear Regression Analysis"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to DataInsightBot - Linear Regression Analysis API",
        "version": "1.0.0",
        "features": [
            "Simple Linear Regression",
            "Multiple Linear Regression", 
            "Automatic Variable Identification",
            "Model Performance Evaluation",
            "Feature Importance Analysis",
            "Residual Analysis",
            "Interactive Visualizations"
        ],
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "DataInsightBot Linear Regression Analysis",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    print("🎯 DataInsightBot - Linear Regression Analysis")
    print("📍 Backend API: http://localhost:8000")
    print("📍 Frontend App: http://localhost:3000")
    print("📍 API Docs: http://localhost:8000/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)