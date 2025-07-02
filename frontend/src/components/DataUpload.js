import React, { useState } from 'react';
import { Upload, Button, Alert, Card } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { dataApi } from '../api/api';

const DataUpload = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);

  const handleUpload = async () => {
    if (!file) {
      setError('请选择一个文件上传');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      console.log('开始上传文件:', file.name);
      const response = await dataApi.uploadData(file);
      console.log('上传成功，响应:', response);
      setUploading(false);
      
      if (onUploadSuccess) {
        onUploadSuccess(response.data);
      }
    } catch (err) {
      console.error('上传错误详情:', {
        message: err.message,
        response: err.response?.data,
        status: err.response?.status,
        headers: err.response?.headers
      });
      
      setUploading(false);
      let errorMessage = '上传失败';
      
      if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message === 'Network Error') {
        errorMessage = '网络连接错误，请检查后端服务是否正常运行';
      } else if (err.code === 'ECONNABORTED') {
        errorMessage = '请求超时，请稍后重试';
      } else {
        errorMessage = err.message || '未知错误';
      }
      
      setError(errorMessage);
    }
  };

  const props = {
    onRemove: () => {
      setFile(null);
    },
    beforeUpload: (file) => {
      // 检查文件类型
      const isCSVOrExcel = 
        file.type === 'text/csv' || 
        file.type === 'application/vnd.ms-excel' || 
        file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
        file.name.endsWith('.csv') ||
        file.name.endsWith('.xlsx') ||
        file.name.endsWith('.xls');
      
      if (!isCSVOrExcel) {
        setError('只支持上传CSV或Excel文件');
        return Upload.LIST_IGNORE;
      }
      
      // 检查文件大小
      const isLessThan10M = file.size / 1024 / 1024 < 10;
      if (!isLessThan10M) {
        setError('文件大小不能超过10MB');
        return Upload.LIST_IGNORE;
      }
      
      setFile(file);
      return false;  // 阻止自动上传
    },
    maxCount: 1
  };

  return (
    <Card title="上传数据文件" style={{ marginBottom: 20 }}>
      <Upload {...props} fileList={file ? [file] : []}>
        <Button icon={<UploadOutlined />}>选择文件</Button>
      </Upload>
      
      {error && (
        <Alert 
          message="上传错误" 
          description={error} 
          type="error" 
          showIcon 
          style={{ marginTop: 16 }} 
        />
      )}
      
      <Button
        type="primary"
        onClick={handleUpload}
        disabled={!file}
        loading={uploading}
        style={{ marginTop: 16 }}
      >
        {uploading ? '上传中...' : '开始上传'}
      </Button>
      
      <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
        支持上传CSV和Excel文件 (最大10MB)
      </div>
    </Card>
  );
};

export default DataUpload;