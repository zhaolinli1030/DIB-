import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 6000000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// 会话API
export const sessionApi = {
  createSession: () => api.post('/session/create'),
  getSession: (sessionId) => api.get(`/session/${sessionId}`),
  submitFeedback: (sessionId, feedback) => api.post(`/session/${sessionId}/feedback`, feedback)
};

// 数据API
export const dataApi = {
  uploadData: (file, sessionId = null) => {
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }
    return api.post('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  },
  getDataSummary: (sessionId) => api.get(`/data/${sessionId}/summary`),
  getDataPreview: (sessionId, rows = 10) => api.get(`/data/${sessionId}/preview?rows=${rows}`),
  getColumnDetails: (sessionId) => api.get(`/data/${sessionId}/columns`)
};

// 分析API
export const analysisApi = {
  submitQuery: (sessionId, query) => api.post('/analysis/${sessionId}/query', { session_id: sessionId, query }),
  getAnalysisResult: (analysisId, sessionId) => api.get(`/analysis/${analysisId}/result?session_id=${sessionId}`),
  getAnalysisSuggestions: (sessionId) => api.get(`/analysis/${sessionId}/suggestions`),
  getAnalysisHistory: (sessionId) => api.get(`/analysis/${sessionId}/history`)
};

export default {
  sessionApi,
  dataApi,
  analysisApi
};