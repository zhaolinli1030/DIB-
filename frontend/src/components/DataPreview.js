import React, { useState, useEffect } from 'react';
import { Table, Card, Spin, Alert, Tabs } from 'antd';
import { dataApi } from '../api/api';

const { TabPane } = Tabs;

const DataPreview = ({ sessionId }) => {
  const [loading, setLoading] = useState(true); // Initialize loading to true
  const [error, setError] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [columnDetails, setColumnDetails] = useState(null);
  const [dataSummary, setDataSummary] = useState(null);

  useEffect(() => {
    if (sessionId) {
      // Reset states when sessionId changes to show loading indicator
      setLoading(true);
      setError(null);
      setPreviewData(null);
      setColumnDetails(null);
      setDataSummary(null);

      // Fetch all data in parallel
      Promise.all([
        dataApi.getDataPreview(sessionId).then(response => setPreviewData(response.data)).catch(err => {
          setError('获取数据预览失败: ' + (err.response?.data?.detail || err.message));
          console.error('Preview error:', err);
        }),
        dataApi.getColumnDetails(sessionId).then(response => setColumnDetails(response.data)).catch(err => {
          // Optionally set a specific error or just log for column details
          console.error('Column details error:', err);
          setError(prevError => prevError || '获取列详情失败: ' + (err.response?.data?.detail || err.message));
        }),
        dataApi.getDataSummary(sessionId).then(response => setDataSummary(response.data)).catch(err => {
          // Optionally set a specific error or just log for data summary
          console.error('Data summary error:', err);
          setError(prevError => prevError || '获取数据摘要失败: ' + (err.response?.data?.detail || err.message));
        })
      ]).finally(() => {
        setLoading(false);
      });
    } else {
      // If no sessionId, clear data and loading state
      setLoading(false);
      setPreviewData(null);
      setColumnDetails(null);
      setDataSummary(null);
      setError(null);
    }
  }, [sessionId]);

  // This function was defined but not used in the original fetch logic,
  // I've integrated its logic into the Promise.all in useEffect.
  // const fetchPreviewData = async () => { ... };
  // const fetchColumnDetails = async () => { ... };
  // const fetchDataSummary = async () => { ... };

  if (loading && !error) { // Show loading only if no error has occurred yet
    return <Spin tip="加载数据预览中..." style={{ display: 'block', marginTop: 20 }} />;
  }

  if (error) {
    return <Alert message="加载数据时出错" description={error} type="error" showIcon style={{ margin: 20 }} />;
  }

  if (!sessionId) {
    return <Card><p>请先上传数据以查看预览和摘要。</p></Card>;
  }
  
  // Dynamic columns for the preview table
  const previewTableColumns = previewData?.columns?.map(columnName => ({
    title: columnName,
    dataIndex: columnName,
    key: columnName,
    ellipsis: true,
    width: 150, // Give a default width
    render: (text) => {
      if (text === null || text === undefined) {
        return <span style={{ color: '#aaa' }}>NULL</span>;
      }
      if (typeof text === 'object') {
        try {
          return JSON.stringify(text);
        } catch (e) {
          return '[Circular Object]';
        }
      }
      return String(text);
    }
  })) || [];

  // Columns for the column details table
  const columnDetailsTableColumns = [
    { title: '列名', dataIndex: 'column', key: 'column', width: 200, ellipsis: true, fixed: 'left' },
    { title: '原始类型', dataIndex: 'type', key: 'type', width: 120 },
    { title: 'LLM语义类型', dataIndex: 'llm_semantic_type', key: 'llm_semantic_type', width: 180, ellipsis: true},
    { title: '唯一值数', dataIndex: 'unique_values', key: 'unique_values', width: 100, sorter: (a, b) => a.unique_values - b.unique_values },
    { title: '缺失值数', dataIndex: 'missing_values', key: 'missing_values', width: 100, sorter: (a, b) => a.missing_values - b.missing_values  },
    { title: '缺失百分比', dataIndex: 'missing_percentage', key: 'missing_percentage', width: 120, sorter: (a, b) => parseFloat(a.missing_percentage) - parseFloat(b.missing_percentage) },
    { title: '均值', dataIndex: 'mean', key: 'mean', width: 120, render: val => val === null || val === undefined ? 'N/A' : val.toFixed(2) },
    { title: '中位数', dataIndex: 'median', key: 'median', width: 120, render: val => val === null || val === undefined ? 'N/A' : val.toFixed(2)  },
    { title: '最小值', dataIndex: 'min', key: 'min', width: 120, render: val => val === null || val === undefined ? 'N/A' : val.toFixed(2)  },
    { title: '最大值', dataIndex: 'max', key: 'max', width: 120, render: val => val === null || val === undefined ? 'N/A' : val.toFixed(2)  },
    { title: 'LLM推断属性', dataIndex: 'llm_inferred_properties', key: 'llm_inferred_properties', width: 300, render: props => Array.isArray(props) ? props.join(', ') : props, ellipsis: true },
    { title: 'LLM质量标记', dataIndex: 'llm_data_quality_flags', key: 'llm_data_quality_flags', width: 300, render: flags => Array.isArray(flags) ? flags.join(', ') : flags, ellipsis: true },
  ];

  const columnDetailsDataSource = columnDetails ? 
    Object.entries(columnDetails).map(([columnName, details]) => ({
        key: columnName, // Use columnName as key for React list
        column: columnName,
        type: details.type,
        llm_semantic_type: details.llm_semantic_type || 'N/A',
        unique_values: details.unique_values,
        missing_values: details.missing_values,
        missing_percentage: details.missing_percentage !== undefined ? `${details.missing_percentage.toFixed(2)}%` : 'N/A',
        mean: details.mean,
        median: details.median,
        min: details.min,
        max: details.max,
        llm_inferred_properties: details.llm_inferred_properties,
        llm_data_quality_flags: details.llm_data_quality_flags,
    })) : [];


  return (
    <Tabs defaultActiveKey="preview" style={{ marginTop: 20 }}>
      <TabPane tab="数据预览" key="preview">
        <Card title={previewData ? `数据预览 (共 ${previewData.total_rows} 行)`: "数据预览"} style={{ marginBottom: 20 }}>
          {previewData && previewData.data && previewData.columns ? (
            <>
              <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
                <Table 
                  dataSource={previewData.data.map((row, index) => ({...row, key: index}))} // Add key to each row
                  columns={previewTableColumns}
                  // rowKey={(record, index) => index} // Using key in dataSource now
                  pagination={{ pageSize: 5, showSizeChanger: false }} // Add simple pagination
                  size="small"
                  bordered
                  scroll={{ x: 'max-content' }} // Enable horizontal scroll
                />
              </div>
              <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
                显示前 {previewData.data.length} 行数据 (总共 {previewData.total_rows} 行)
              </div>
            </>
          ) : (
            <p>暂无数据预览。</p>
          )}
        </Card>
      </TabPane>
      
      <TabPane tab="列详情与统计" key="columns">
        <Card title="列详情与统计" style={{ marginBottom: 20 }}>
          {columnDetails ? (
            <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
              <Table
                dataSource={columnDetailsDataSource}
                columns={columnDetailsTableColumns}
                rowKey="column"
                size="small"
                bordered
                scroll={{ x: 'max-content' }} // Enable horizontal scroll
                pagination={{ pageSize: 10, showSizeChanger: true, pageSizeOptions: ['5', '10', '20'] }}
              />
            </div>
          ) : (
            <p>暂无列详情信息。</p>
          )}
        </Card>
      </TabPane>
      
      <TabPane tab="数据质量报告" key="quality">
        <Card title="数据质量报告" style={{ marginBottom: 20 }}>
          {dataSummary && dataSummary.quality_analysis ? (
            <div>
              <h3>缺失值统计</h3>
              <p>总缺失值数量: {dataSummary.quality_analysis.missing_values?.total ?? 'N/A'}</p>
              <p>总体缺失值百分比: {dataSummary.quality_analysis.missing_values?.percentage !== undefined ? dataSummary.quality_analysis.missing_values.percentage.toFixed(2) + '%' : 'N/A'}</p>
              {dataSummary.quality_analysis.missing_values?.by_column && Object.keys(dataSummary.quality_analysis.missing_values.by_column).length > 0 && (
                <>
                  <h4>按列缺失详情:</h4>
                  <ul>
                    {Object.entries(dataSummary.quality_analysis.missing_values.by_column).map(([col, count]) => (
                      <li key={col}>{col}: {count} 个缺失值</li>
                    ))}
                  </ul>
                </>
              )}
              
              <h3 style={{marginTop: 16}}>重复行统计</h3>
              <p>总重复行数: {dataSummary.quality_analysis.duplicate_rows?.total ?? 'N/A'}</p>
              <p>重复行百分比: {dataSummary.quality_analysis.duplicate_rows?.percentage !== undefined ? dataSummary.quality_analysis.duplicate_rows.percentage.toFixed(2) + '%' : 'N/A'}</p>
              
              <h3 style={{marginTop: 16}}>潜在异常值 (IQR方法)</h3>
              {/* 修改这里以匹配后端的 'potential_outliers_iqr' 键名和结构 */}
              {dataSummary.quality_analysis.potential_outliers_iqr && 
               typeof dataSummary.quality_analysis.potential_outliers_iqr === 'object' && 
               Object.keys(dataSummary.quality_analysis.potential_outliers_iqr).length > 0 ? (
                <ul>
                  {Object.entries(dataSummary.quality_analysis.potential_outliers_iqr).map(([column, outlierDetails]) => (
                    <li key={column}>
                      <strong>{column}:</strong> {outlierDetails.count} 个潜在异常值 
                      (占该列有效数据 {outlierDetails.percentage !== undefined ? outlierDetails.percentage.toFixed(1) : 'N/A'}%)
                      {/* , 范围: [{outlierDetails.lower_bound !== undefined ? outlierDetails.lower_bound.toFixed(2) : 'N/A'}, {outlierDetails.upper_bound !== undefined ? outlierDetails.upper_bound.toFixed(2) : 'N/A'}] */}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>未检测到或无法显示潜在异常值信息。</p>
              )}
            </div>
          ) : (
            <p>暂无数据质量报告。</p>
          )}
        </Card>
      </TabPane>
    </Tabs>
  );
};

export default DataPreview;
