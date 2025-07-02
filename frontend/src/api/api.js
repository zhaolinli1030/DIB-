import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 600000,  // 10 minutes
  headers: {
    'Content-Type': 'application/json'
  }
});

// 会话API
export const sessionApi = {
  createSession: () => api.post('/api/session/create'),
  getSession: (sessionId) => api.get(`/api/session/${sessionId}`),
  submitFeedback: (sessionId, feedback) => api.post(`/api/session/${sessionId}/feedback`, feedback)
};

// 数据API
export const dataApi = {
  uploadData: (file, sessionId = null) => {
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }
    return api.post('/api/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 3000000  // 增加到50分钟超时
    });
  },
  getDataSummary: (sessionId) => api.get(`/api/data/${sessionId}/summary`),
  getDataPreview: (sessionId, rows = 10) => api.get(`/api/data/${sessionId}/preview?rows=${rows}`),
  getColumnDetails: (sessionId) => api.get(`/api/data/${sessionId}/columns`)
};

// 分析API
export const analysisApi = {
  submitQuery: (sessionId, query) => api.post(`/api/analysis/${sessionId}/query`, { query }, {
    timeout: 3000000  // 增加到50分钟超时
  }),
  getAnalysisResult: (analysisId, sessionId) => api.get(`/api/analysis/${analysisId}/result?session_id=${sessionId}`),
  getAnalysisSuggestions: (sessionId) => api.get(`/api/analysis/${sessionId}/suggestions`),
  getAnalysisHistory: (sessionId) => api.get(`/api/analysis/${sessionId}/history`)
};

const apiExports = {
  sessionApi,
  dataApi,
  analysisApi
};

export default apiExports;