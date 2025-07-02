import React, { useState, useEffect, useCallback } from 'react';
import { Input, Button, Card, Tag, Spin, Alert, Progress } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { analysisApi } from '../api/api';

const { TextArea } = Input;

const AnalysisQuery = ({ sessionId, onAnalysisResult }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');

  const fetchSuggestions = useCallback(async () => {
    try {
      const response = await analysisApi.getAnalysisSuggestions(sessionId);
      setSuggestions(response.data.suggestions || []);
    } catch (err) {
      console.error('Suggestions error:', err);
    }
  }, [sessionId]);

  useEffect(() => {
    if (sessionId) {
      fetchSuggestions();
    }
  }, [sessionId, fetchSuggestions]);

  const simulateProgress = () => {
    const steps = [
      { percent: 20, text: '正在理解数据分析需求...' },
      { percent: 40, text: '正在识别目标变量和特征变量...' },
      { percent: 60, text: '正在训练线性回归模型...' },
      { percent: 80, text: '正在生成模型性能分析...' },
      { percent: 95, text: '正在生成可视化图表...' }
    ];
    
    let currentStep = 0;
    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        setProgress(steps[currentStep].percent);
        setProgressText(steps[currentStep].text);
        currentStep++;
      } else {
        clearInterval(interval);
      }
    }, 30000); // 每30秒更新一次进度
    
    return interval;
  };

  const handleSubmit = async () => {
    if (!query.trim()) {
      setError('请输入数据分析查询');
      return;
    }

    setLoading(true);
    setError(null);
    setProgress(10);
    setProgressText('开始处理数据分析...');
    
    const progressInterval = simulateProgress();

    try {
      const response = await analysisApi.submitQuery(sessionId, query);
      clearInterval(progressInterval);
      setLoading(false);
      setProgress(100);
      setProgressText('数据分析完成！');
      
      setTimeout(() => {
        setProgress(0);
        setProgressText('');
      }, 2000);
      
      if (onAnalysisResult) {
        const analysisResult = {
          analysis_id: response.data.analysis_id,
          ...response.data.result
        };
        onAnalysisResult(analysisResult);
      }
    } catch (err) {
      clearInterval(progressInterval);
      setLoading(false);
      setProgress(0);
      setProgressText('');
      
      let errorMessage = '数据分析失败，请重试';
      if (err.code === 'ECONNABORTED') {
        errorMessage = '分析处理时间较长，请稍后重试。如果数据量较大或查询复杂，处理时间可能需要几分钟。';
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      }
      
      setError(errorMessage);
      console.error('Analysis error:', err);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setQuery(suggestion.description);
  };

  // 默认的回归分析建议
  const defaultSuggestions = [
    {
      description: "分析哪些因素影响销售额？",
      type: "multiple_regression"
    },
    {
      description: "预测房价基于面积和位置",
      type: "multiple_regression"
    },
    {
      description: "研究教育水平对收入的影响",
      type: "simple_regression"
    }
  ];

  // 只保留回归分析相关建议
  const regressionTypes = ['simple_regression', 'multiple_regression'];
  const filteredSuggestions = (suggestions.length > 0
    ? suggestions.filter(s => regressionTypes.includes(s.type) || (s.description && s.description.includes('回归')))
    : defaultSuggestions);

  const displaySuggestions = filteredSuggestions;

  return (
    <Card title="数据分析查询" style={{ marginBottom: 20 }}>
      <div style={{ marginBottom: 16, padding: 12, backgroundColor: '#f6ffed', borderRadius: 6 }}>
        <h4 style={{ margin: '0 0 8px 0', color: '#52c41a' }}>💡 数据分析说明</h4>
        <p style={{ margin: 0, fontSize: 14, color: '#666' }}>
          数据分析用于分析变量之间的线性关系，预测一个变量（目标变量）基于其他变量（特征变量）的值。
          支持简单线性回归（一个特征）和多元线性回归（多个特征）。
        </p>
      </div>

      <TextArea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="请用自然语言描述您想要进行的数据分析，例如：'分析哪些因素影响销售额？' 或 '预测房价基于面积和位置'"
        autoSize={{ minRows: 3, maxRows: 6 }}
        style={{ marginBottom: 16 }}
        disabled={loading}
      />
      
      <Button
        type="primary"
        icon={<SearchOutlined />}
        onClick={handleSubmit}
        loading={loading}
        disabled={loading}
      >
        {loading ? '分析中...' : '开始数据分析'}
      </Button>
      
      {loading && (
        <div style={{ marginTop: 16 }}>
          <Spin tip={progressText || '正在执行数据分析，请耐心等待...'} />
          <Progress 
            percent={progress} 
            status="active" 
            style={{ marginTop: 8 }}
            strokeColor={{
              '0%': '#108ee9',
              '100%': '#87d068',
            }}
          />
          <div style={{ fontSize: 12, color: '#666', marginTop: 8 }}>
            💡 提示：数据分析可能需要1-3分钟时间，请耐心等待
          </div>
        </div>
      )}
      
      {error && (
        <Alert 
          message="查询错误" 
          description={error} 
          type="error" 
          showIcon 
          style={{ marginTop: 16 }}
          action={
            <Button size="small" onClick={() => setError(null)}>
              知道了
            </Button>
          }
        />
      )}
      
      {displaySuggestions.length > 0 && !loading && (
        <div style={{ marginTop: 16 }}>
          <h4>数据分析建议:</h4>
          <div>
            {displaySuggestions.map((suggestion, index) => (
              <Tag 
                key={index} 
                color={suggestion.type === 'simple_regression' ? 'green' : 'blue'} 
                style={{ margin: '4px', cursor: 'pointer' }} 
                onClick={() => handleSuggestionClick(suggestion)}
              >
                {suggestion.description}
              </Tag>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
};

export default AnalysisQuery;