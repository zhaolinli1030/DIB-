import React, { useState, useEffect } from 'react';
import { Layout, Steps, Button, message, Typography } from 'antd';
import { UploadOutlined, DatabaseOutlined, LineChartOutlined } from '@ant-design/icons';
import DataUpload from './components/DataUpload';
import DataPreview from './components/DataPreview';
import AnalysisQuery from './components/AnalysisQuery';
import AnalysisResult from './components/AnalysisResult';
import { sessionApi } from './api/api';

const { Header, Content, Footer } = Layout;
const { Step } = Steps;
const { Title } = Typography;

const App = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  useEffect(() => {
    createSession();
  }, []);

  const createSession = async () => {
    try {
      const response = await sessionApi.createSession();
      setSessionId(response.data.session_id);
    } catch (err) {
      console.error('Session creation error:', err);
      message.error('创建会话失败，请刷新页面重试');
    }
  };

  const handleUploadSuccess = (result) => {
    setUploadResult(result);
    setSessionId(result.session_id);
    message.success('文件上传成功');
    setCurrentStep(1);
  };

  const handleAnalysisResult = (result) => {
    setAnalysisResult(result);
    message.success('数据分析完成');
    setCurrentStep(2);
  };

  const steps = [
    {
      title: '上传数据',
      content: <DataUpload onUploadSuccess={handleUploadSuccess} />,
      icon: <UploadOutlined />
    },
    {
      title: '数据分析',
      content: (
        <>
          {uploadResult && (
            <div style={{ marginBottom: 20, padding: 16, background: '#f6ffed', borderRadius: 6 }}>
              <strong>📊 已上传文件：</strong>{uploadResult.filename} 
              <span style={{ marginLeft: 20 }}>
                <strong>📈 数据规模：</strong>{uploadResult.rows} 行 × {uploadResult.columns} 列
              </span>
              <div style={{ marginTop: 8, fontSize: 14, color: '#52c41a' }}>
                💡 系统将自动识别数值型变量，支持简单和多元线性回归分析
              </div>
            </div>
          )}
          <DataPreview sessionId={sessionId} />
          <AnalysisQuery sessionId={sessionId} onAnalysisResult={handleAnalysisResult} />
        </>
      ),
      icon: <DatabaseOutlined />
    },
    {
      title: '分析结果',
      content: (
        <>
          <AnalysisResult 
            analysisId={analysisResult?.analysis_id} 
            sessionId={sessionId} 
          />
          <Button 
            type="primary" 
            onClick={() => setCurrentStep(1)} 
            style={{ marginBottom: 20 }}
          >
            进行新的数据分析
          </Button>
        </>
      ),
      icon: <LineChartOutlined />
    }
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 20px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
          <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
            📊 DataInsightBot
          </Title>
          <div style={{ marginLeft: 20, fontSize: 16, color: '#52c41a', fontWeight: 'bold' }}>
            数据分析助手
          </div>
          <div style={{ marginLeft: 20, fontSize: 14, color: '#888' }}>
            
          </div>
        </div>
      </Header>
      
      <Content style={{ padding: '20px 50px', background: '#fafafa' }}>
        <div style={{ background: '#fff', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
          <Steps current={currentStep} style={{ marginBottom: 30 }}>
            {steps.map(item => (
              <Step key={item.title} title={item.title} icon={item.icon} />
            ))}
          </Steps>
          
          <div className="steps-content">
            {steps[currentStep].content}
          </div>
        </div>
      </Content>
      
      <Footer style={{ textAlign: 'center', background: '#f0f2f5' }}>
        <div>
          <strong>DataInsightBot</strong> ©{new Date().getFullYear()} - 专注于数据分析的智能工具
        </div>
        <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
          支持简单线性回归 • 多元线性回归 • 自动变量识别 • 模型性能评估
        </div>
      </Footer>
    </Layout>
  );
};

export default App;