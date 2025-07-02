import React, { useEffect, useState, useCallback } from 'react';
import { Card, Tabs, Typography, Spin, Alert, List } from 'antd';
import ReactECharts from 'echarts-for-react';
import { analysisApi } from '../api/api';

const { TabPane } = Tabs;
const { Title, Paragraph, Text } = Typography;

// 错误边界组件
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('AnalysisResult Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Alert
          message="组件渲染错误"
          description={`渲染过程中发生错误: ${this.state.error?.message || '未知错误'}`}
          type="error"
          showIcon
        />
      );
    }
    return this.props.children;
  }
}

const AnalysisResult = ({ analysisId, sessionId }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const fetchAnalysisResult = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('Fetching analysis result:', { analysisId, sessionId });
      const response = await analysisApi.getAnalysisResult(analysisId, sessionId);
      console.log('Analysis result response:', response.data);
      setResult(response.data);
    } catch (err) {
      console.error('Analysis result error:', err);
      setError('获取分析结果失败: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  }, [analysisId, sessionId]);

  useEffect(() => {
    console.log('AnalysisResult useEffect:', { analysisId, sessionId });
    if (analysisId && sessionId) {
      fetchAnalysisResult();
    } else {
      console.warn('Missing analysisId or sessionId:', { analysisId, sessionId });
    }
  }, [analysisId, sessionId, fetchAnalysisResult]);

  if (loading) {
    return <Spin tip="加载数据分析结果中..." />;
  }

  if (error) {
    return (
      <Alert 
        message="错误" 
        description={
          <div>
            <p>{error}</p>
            <p>调试信息: analysisId={analysisId}, sessionId={sessionId}</p>
          </div>
        } 
        type="error" 
        showIcon 
      />
    );
  }

  if (!result) {
    return (
      <Alert 
        message="暂无分析结果" 
        description="请先执行数据分析查询"
        type="info" 
        showIcon 
      />
    );
  }

  // 深度清理函数：递归检查并清理所有React元素
  const deepCleanReactElements = (obj, depth = 0) => {
    if (depth > 10) {
      console.warn('Recursive depth limit reached, returning safe value');
      return '[Deep Object]';
    }
    
    if (obj === null || obj === undefined) return obj;
    
    if (typeof obj === 'object' && (
      obj.$$typeof || 
      obj._owner || 
      obj._store || 
      (obj.type && obj.props) ||
      (obj.key !== undefined && obj.ref !== undefined)
    )) {
      console.warn('Found React element, converting to string:', obj);
      return '[React Element]';
    }
    
    if (typeof obj === 'function') {
      console.warn('Found function, converting to string:', obj.name || 'anonymous');
      return '[Function]';
    }
    
    if (Array.isArray(obj)) {
      return obj.map(item => deepCleanReactElements(item, depth + 1));
    }
    
    if (typeof obj === 'object') {
      try {
        const cleaned = {};
        for (const [key, value] of Object.entries(obj)) {
          if (key.startsWith('_') || key.startsWith('$$')) {
            continue;
          }
          cleaned[key] = deepCleanReactElements(value, depth + 1);
        }
        return cleaned;
      } catch (error) {
        console.warn('Error cleaning object, returning safe string:', error);
        return '[Object Error]';
      }
    }
    
    return obj;
  };

  // 清理所有数据
  const cleanedResult = deepCleanReactElements(result);
  const cleanedInterpretation = cleanedResult.interpretation || {};
  const cleanedSummary = String(cleanedInterpretation.summary || "无可用解释");
  const cleanedInsights = Array.isArray(cleanedInterpretation.insights) ? cleanedInterpretation.insights : [];
  const cleanedNextSteps = Array.isArray(cleanedInterpretation.next_steps) ? cleanedInterpretation.next_steps : [];

  // 安全的文本渲染函数
  const safeText = (value, fallback = '') => {
    if (value === null || value === undefined) return fallback;
    
    if (typeof value === 'object' && (
      value.$$typeof || 
      value._owner || 
      value._store ||
      (value.type && value.props)
    )) {
      console.warn('Attempted to render React element as text:', value);
      return '[React Element - Not Renderable]';
    }
    
    return String(value);
  };

  // 递归搜索对象树中的模型信息
  function findModelInfo(obj, depth = 0) {
    if (!obj || typeof obj !== 'object' || depth > 5) return null;
    
    // 直接检查当前层级
    if (obj.model_info) return obj.model_info;
    
    // 递归搜索所有子对象
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'object' && value !== null) {
        const result = findModelInfo(value, depth + 1);
        if (result) return result;
      }
    }
    
    return null;
  }
  
  // 递归搜索对象树中的性能指标
  function findPerf(obj, depth = 0) {
    if (!obj || typeof obj !== 'object' || depth > 5) return null;
    
    // 直接检查当前层级
    if (obj.performance_metrics) return obj.performance_metrics;
    
    // 递归搜索所有子对象
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'object' && value !== null) {
        const result = findPerf(value, depth + 1);
        if (result) return result;
      }
    }
    
    return null;
  }
const modelInfo = findModelInfo(cleanedResult);
const perf = findPerf(cleanedResult);

  // 渲染图表
  const renderCharts = () => {
    const visualization = cleanedResult.visualization;
    
    if (!visualization || !visualization.charts || visualization.charts.length === 0) {
      return (
        <Alert 
          message="暂无可视化图表" 
          description="分析结果中没有生成可视化图表"
          type="info" 
          showIcon 
        />
      );
    }

    return (
      <div>
        <Title level={4}>可视化图表</Title>
        {visualization.charts.map((chartConfig, index) => (
          <ErrorBoundary key={index}>
            <Card 
              title={chartConfig.title || `图表 ${index + 1}`} 
              style={{ marginBottom: 16 }}
            >
              <ReactECharts 
                option={convertToEChartsOptions(chartConfig)} 
                style={{ height: '400px' }}
                opts={{ renderer: 'canvas' }}
              />
              {chartConfig.description && (
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary">{chartConfig.description}</Text>
                </div>
              )}
            </Card>
          </ErrorBoundary>
        ))}
      </div>
    );
  };

  // 转换图表配置为ECharts选项
  const convertToEChartsOptions = (chartConfig) => {
    const { type, data, xField, yField, color, colorField, angleField } = chartConfig;
    
      const baseOptions = {
        title: { 
        text: chartConfig.title || '',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis'
      },
        grid: { 
        left: '3%',
        right: '4%',
        bottom: '3%',
          containLabel: true 
        }
      };
      
      switch (type) {
      case 'bar':
        return createBarChart(baseOptions, data, xField, yField, color);
        case 'line':
        return createLineChart(baseOptions, data, xField, yField, color);
      case 'scatter':
        return createScatterChart(baseOptions, data, xField, yField, color, chartConfig.title);
      case 'box':
        return createBoxChart(baseOptions, data, xField, yField, color);
        case 'pie':
        return createPieChart(baseOptions, data, angleField, colorField);
      case 'table':
        return createTableChart(baseOptions, data, chartConfig.columns);
        default:
        return {
          ...baseOptions,
          series: [{
            type: 'bar',
            data: data || []
          }]
        };
    }
  };

  const createBarChart = (baseOptions, data, xField, yField, color) => ({
        ...baseOptions,
        xAxis: {
          type: 'category',
      data: data?.map(item => item[xField]) || []
        },
        yAxis: {
          type: 'value'
        },
        series: [{
      type: 'bar',
      data: data?.map(item => item[yField]) || [],
      itemStyle: {
        color: color || '#5B8FF9'
      }
    }]
  });

  const createLineChart = (baseOptions, data, xField, yField, color) => ({
        ...baseOptions,
        xAxis: {
          type: 'category',
      data: data?.map(item => item[xField]) || []
        },
        yAxis: {
          type: 'value'
        },
        series: [{
      type: 'line',
      data: data?.map(item => item[yField]) || [],
          itemStyle: {
        color: color || '#5B8FF9'
      }
    }]
  });

  const createScatterChart = (baseOptions, data, xField, yField, color, chartTitle) => {
    // 检查是否是预测值vs实际值图表
    const isPredictionChart = chartTitle && (
      chartTitle.includes('预测值') && chartTitle.includes('实际值') ||
      chartTitle.includes('predicted') && chartTitle.includes('actual')
    );
    
    const series = [{
      type: 'scatter',
      data: data?.map(item => [item[xField], item[yField]]) || [],
            itemStyle: {
        color: color || '#5B8FF9'
      }
    }];
    
    // 如果是预测值vs实际值图表，添加y=x参考线
    if (isPredictionChart && data && data.length > 0) {
      // 计算数据范围
      const xValues = data.map(item => item[xField]).filter(val => val !== null && val !== undefined);
      const yValues = data.map(item => item[yField]).filter(val => val !== null && val !== undefined);
      
      if (xValues.length > 0 && yValues.length > 0) {
        const minVal = Math.min(Math.min(...xValues), Math.min(...yValues));
        const maxVal = Math.max(Math.max(...xValues), Math.max(...yValues));
        
        // 添加y=x参考线
        series.push({
          type: 'line',
          data: [[minVal, minVal], [maxVal, maxVal]],
          lineStyle: {
            color: '#FF6B6B',
            type: 'dashed',
            width: 2
          },
          symbol: 'none',
          name: '理想预测线 (y=x)'
        });
      }
    }
      
      return {
      ...baseOptions,
        tooltip: {
        trigger: 'item',
          formatter: function(params) {
          if (params.seriesType === 'scatter') {
            const x = params.data[0];
            const y = params.data[1];
            return `(${x}, ${y})`;
          }
          return params.name;
        }
        },
        xAxis: {
        type: 'value'
        },
        yAxis: {
        type: 'value'
      },
      series: series
    };
  };

  const createPieChart = (baseOptions, data, angleField, colorField) => ({
    ...baseOptions,
    series: [{
      type: 'pie',
      radius: '50%',
      data: data?.map(item => ({
        name: item[colorField],
        value: item[angleField]
      })) || []
    }]
  });

  const createTableChart = (baseOptions, data, columns) => ({
    ...baseOptions,
    series: [{
      type: 'table',
      data: data || [],
      columns: columns || []
    }]
  });

  const createBoxChart = (baseOptions, data, xField, yField, color) => ({
    ...baseOptions,
    xAxis: {
          type: 'category',
      data: data?.map(item => item[xField]) || []
    },
    yAxis: {
      type: 'value'
        },
        series: [{
      type: 'boxplot',
      data: data?.map(item => [item.min, item.q1, item.median, item.q3, item.max]) || [],
            itemStyle: {
        color: color || '#52C41A'
      }
    }]
  });

  return (
    <div>
      <Title level={3}>数据分析结果</Title>
      <Tabs defaultActiveKey="charts">
        <TabPane tab="可视化图表" key="charts">
          {renderCharts()}
        </TabPane>
        <TabPane tab="分析解释" key="interpretation">
          <Card title="模型摘要" style={{ marginBottom: 16 }}>
            {modelInfo && (
              <div style={{ marginBottom: 12 }}>
                <div><b>截距：</b> {modelInfo.intercept?.toFixed(4)}</div>
                
                {/* 获取编码信息 */}
                {(() => {
                  const encodingInfo = modelInfo.encoding_info || {};
                  const categoricalEncoding = encodingInfo.categorical_encoding || {};
                  const featureImportance = modelInfo.feature_importance || {};
                  
                  // 按特征类型分组
                  const numericalFeatures = [];
                  const categoricalFeatures = {};
                  const interactionFeatures = [];
                  
                  Object.entries(featureImportance).forEach(([feature, info]) => {
                    const featureType = info.feature_type || 'numerical';
                    const originalFeature = info.original_feature || feature;
                    
                    if (featureType === 'numerical') {
                      numericalFeatures.push([feature, info]);
                    } else if (featureType === 'categorical_dummy') {
                      if (!categoricalFeatures[originalFeature]) {
                        categoricalFeatures[originalFeature] = [];
                      }
                      categoricalFeatures[originalFeature].push([feature, info]);
                    } else if (featureType === 'interaction') {
                      interactionFeatures.push([feature, info]);
                    }
                  });
                  
                  return (
                    <div>
                      {/* 数值变量 */}
                      {numericalFeatures.length > 0 && (
                        <div style={{ marginBottom: 16 }}>
                          <div style={{ fontWeight: 'bold', marginBottom: 8, color: '#1890ff' }}>数值变量:</div>
                          {numericalFeatures.map(([feature, info]) => (
                            <div key={feature} style={{ marginBottom: 8, marginLeft: 16 }}>
                              <div><b>{feature}：</b> 系数 = {info.coefficient?.toFixed(4)}</div>
                              <div style={{ fontSize: 12, color: '#666', marginLeft: 16 }}>
                                <span>p值 = {info.p_value?.toFixed(4)} {info.p_value < 0.001 ? '***' : info.p_value < 0.01 ? '**' : info.p_value < 0.05 ? '*' : ''}</span>
                                <br />
                                <span>相关性: {info.relevance}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {/* 分类变量 */}
                      {Object.keys(categoricalFeatures).length > 0 && (
                        <div style={{ marginBottom: 16 }}>
                          <div style={{ fontWeight: 'bold', marginBottom: 8, color: '#52c41a' }}>分类变量:</div>
                          {Object.entries(categoricalFeatures).map(([originalCat, dummyList]) => {
                            const referenceCategory = categoricalEncoding[originalCat]?.reference_category || '未知';
                            return (
                              <div key={originalCat} style={{ marginBottom: 12, marginLeft: 16 }}>
                                <div style={{ fontWeight: 'bold' }}>
                                  {originalCat} <span style={{ fontWeight: 'normal', color: '#888' }}>(参考类别: {referenceCategory})</span>
                                </div>
                                {dummyList.map(([dummyFeature, info]) => {
                                  const categoryName = dummyFeature.replace(`${originalCat}_`, '');
                          return (
                                    <div key={dummyFeature} style={{ marginBottom: 6, marginLeft: 16 }}>
                                      <div><b>{categoryName}：</b> 系数 = {info.coefficient?.toFixed(4)}</div>
                                      <div style={{ fontSize: 12, color: '#666', marginLeft: 16 }}>
                                        <span>p值 = {info.p_value?.toFixed(4)} {info.p_value < 0.001 ? '***' : info.p_value < 0.01 ? '**' : info.p_value < 0.05 ? '*' : ''}</span>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            );
                          })}
                        </div>
                      )}
                      
                      {/* 交互项 */}
                      {interactionFeatures.length > 0 && (
                        <div style={{ marginBottom: 16 }}>
                          <div style={{ fontWeight: 'bold', marginBottom: 8, color: '#fa8c16' }}>交互项:</div>
                          {interactionFeatures.map(([feature, info]) => (
                            <div key={feature} style={{ marginBottom: 8, marginLeft: 16 }}>
                              <div><b>{feature}：</b> 系数 = {info.coefficient?.toFixed(4)}</div>
                              <div style={{ fontSize: 12, color: '#666', marginLeft: 16 }}>
                                <span>p值 = {info.p_value?.toFixed(4)} {info.p_value < 0.001 ? '***' : info.p_value < 0.01 ? '**' : info.p_value < 0.05 ? '*' : ''}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {/* 显著性说明 */}
                      <div style={{ fontSize: 11, color: '#888', marginTop: 12, fontStyle: 'italic' }}>
                        显著性标记: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05
                      </div>
                      
                      {/* 完整回归方程 */}
                      {(() => {
                        const targetColumn = modelInfo.target_column || 'Y';
                        const intercept = modelInfo.intercept || 0;
                        const featureImportance = modelInfo.feature_importance || {};
                        
                        // 构建方程项
                        const equationTerms = [];
                        
                        // 添加截距项
                        if (Math.abs(intercept) > 0.0001) {
                          equationTerms.push(intercept >= 0 ? `${intercept.toFixed(3)}` : `${intercept.toFixed(3)}`);
                        }
                        
                        // 添加各个特征项
                        Object.entries(featureImportance).forEach(([feature, info]) => {
                          const coeff = info.coefficient || 0;
                          if (Math.abs(coeff) > 0.0001) {
                            const coeffStr = Math.abs(coeff).toFixed(3);
                            const sign = coeff >= 0 ? '+' : '-';
                            const term = `${sign} ${coeffStr} × ${feature}`;
                            equationTerms.push(term);
                          }
                        });
                        
                        // 如果没有有效项，至少显示截距
                        if (equationTerms.length === 0) {
                          equationTerms.push(intercept.toFixed(3));
                        }
                        
                        // 构建完整方程
                        let equation = `${targetColumn} = `;
                        if (equationTerms.length > 0) {
                          // 处理第一项（去掉正号）
                          let firstTerm = equationTerms[0];
                          if (firstTerm.startsWith('+ ')) {
                            firstTerm = firstTerm.substring(2);
                          }
                          equation += firstTerm;
                          
                          // 添加其余项
                          for (let i = 1; i < equationTerms.length; i++) {
                            equation += ` ${equationTerms[i]}`;
                          }
                        }
                        
                        return (
                          <div style={{ 
                            marginTop: 16, 
                            padding: 12, 
                            backgroundColor: '#f0f2f5', 
                            borderRadius: 6,
                            border: '1px solid #d9d9d9'
                          }}>
                            <div style={{ fontWeight: 'bold', marginBottom: 8, color: '#1890ff' }}>
                              完整回归方程:
                            </div>
                            <div style={{ 
                              fontFamily: 'Monaco, Consolas, monospace', 
                              fontSize: 14,
                              color: '#262626',
                              wordBreak: 'break-all'
                            }}>
                              {equation}
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  );
                })()}
              </div>
            )}
            {perf && (
              <div>
                <div><b>R²：</b> {(perf.r2 ?? perf.train_r2 ?? perf.test_r2)?.toFixed(4)} <span style={{ fontSize: 14, color: '#666' }}>(决定系数，数值越接近1表示模型拟合越好，但过高可能存在过拟合风险)</span></div>
              </div>
            )}
            {(!modelInfo && !perf) && <div style={{color:'#888'}}>未找到模型摘要信息</div>}
          </Card>
          <Card title="分析总结">
            <Paragraph>{safeText(cleanedSummary)}</Paragraph>
          </Card>
          {cleanedInsights.length > 0 && (
            <Card title="关键洞察" style={{ marginTop: 16 }}>
              <List
                dataSource={cleanedInsights}
                renderItem={(insight, index) => (
                  <List.Item>
                    <div>
                      <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
                        {index + 1}. {insight.title || `洞察 ${index + 1}`}
                      </div>
                      <div style={{ color: '#666', fontSize: 14 }}>
                        {insight.description || safeText(insight)}
                      </div>
                    </div>
                  </List.Item>
                )}
              />
            </Card>
          )}
          {cleanedNextSteps.length > 0 && (
            <Card title="建议后续步骤" style={{ marginTop: 16 }}>
              <List
                dataSource={cleanedNextSteps}
                renderItem={(step, index) => (
                  <List.Item>
                    <div>
                      <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
                        {index + 1}. {step.title || `步骤 ${index + 1}`}
                      </div>
                      <div style={{ color: '#666', fontSize: 14 }}>
                        {step.description || safeText(step)}
                      </div>
                    </div>
                          </List.Item>
                )}
              />
            </Card>
          )}
          </TabPane>
        </Tabs>
    </div>
  );
};

export default AnalysisResult;