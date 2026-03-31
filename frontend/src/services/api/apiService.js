/**
 * API Service for CareerPath AI
 * Handles communication with the FastAPI backend
 */

// API base URL resolution (in priority order):
// 1. VITE_API_URL env var — set this in Render/Vercel/etc. to the backend URL
//    e.g. https://careerpath-ai-backend-4d7q.onrender.com
// 2. localhost dev — direct to local FastAPI server
// 3. Same-origin production (Docker/nginx proxy where backend serves /api)
const _envApiUrl = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace(/\/$/, '')  // strip any trailing slash
  : null;

const API_BASE_URL = _envApiUrl
  ?? (window.location.hostname === 'localhost'
    ? 'http://localhost:8000'   // Development mode (Vite)
    : '');                      // Same-origin production (Docker/nginx proxy)

class ApiService {
  /**
   * Predict career path from uploaded resume
   * @param {File} file - Resume file (PDF)
   * @returns {Promise} Prediction results
   */
  async predictCareerPath(file) {
    if (!file) {
      throw new Error('No file provided');
    }

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      throw new Error('Only PDF files are supported');
    }

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      throw new Error(`File size exceeds 10MB. Your file is ${(file.size / (1024 * 1024)).toFixed(2)}MB`);
    }

    // Create form data
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = 'Prediction failed';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          // Response body is not JSON or empty
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      if (error.message.includes('Failed to fetch')) {
        throw new Error('Cannot connect to the server. Please ensure the backend is running on http://localhost:8000');
      }
      throw error;
    }
  }

  /**
   * Get list of available career paths
   * @returns {Promise} List of careers
   */
  async getAvailableCareers() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/careers`);

      if (!response.ok) {
        let errorMessage = 'Failed to fetch careers';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching careers:', error);
      throw error;
    }
  }

  /**
   * Generate personalized learning roadmap using Gemini LLM
   * @param {string} careerPath - The predicted career path
   * @param {string} resumeText - Raw resume text
   * @returns {Promise} Learning roadmap data
   */
  async generateLearningRoadmap(careerPath, resumeText) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/learning-roadmap`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          career_path: careerPath,
          resume_text: resumeText,
        }),
      });

      if (!response.ok) {
        let errorMessage = 'Failed to generate learning roadmap';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error generating learning roadmap:', error);
      throw error;
    }
  }

  /**
   * Generate personalized certification recommendations using Gemini LLM
   * @param {string} careerPath - The predicted career path
   * @param {string} resumeText - Raw resume text
   * @returns {Promise} Certification recommendations data
   */
  async generateCertifications(careerPath, resumeText) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/certifications`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          career_path: careerPath,
          resume_text: resumeText,
        }),
      });

      if (!response.ok) {
        let errorMessage = 'Failed to generate certifications';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error generating certifications:', error);
      throw error;
    }
  }

  /**
   * Search for jobs via backend JSearch proxy
   * @param {Object} params - Job search parameters
   * @returns {Promise} Job search results
   */
  async searchJobs(params) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        let errorMessage = 'Failed to search jobs';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error searching jobs:', error);
      throw error;
    }
  }

  /**
   * Health check
   * @returns {Promise} Health status
   */
  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);

      if (!response.ok) {
        let errorMessage = 'Health check failed';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }
  /**
   * Parse FastAPI error detail — handles both string and Pydantic array responses.
   * e.g. [{loc:[...], msg:"...", type:"..."}]  →  "Password must be at least 6 characters"
   */
  _parseDetail(detail, fallback) {
    if (!detail) return fallback;
    if (Array.isArray(detail)) {
      return detail.map(e => e.msg || JSON.stringify(e)).join('. ');
    }
    return String(detail);
  }

  /**
   * Register a new user
   * @param {string} name
   * @param {string} email
   * @param {string} password
   * @returns {Promise<{success, token, user}>}
   */
  async registerUser(name, email, password) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: name, email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(this._parseDetail(data.detail, 'Registration failed'));
      }

      return data;
    } catch (error) {
      if (error.message.includes('Failed to fetch')) {
        throw new Error('Cannot connect to the server. Please ensure the backend is running.');
      }
      throw error;
    }
  }

  /**
   * Login with email and password
   * @param {string} email
   * @param {string} password
   * @returns {Promise<{success, token, user}>}
   */
  async loginUser(email, password) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(this._parseDetail(data.detail, 'Login failed'));
      }

      return data;
    } catch (error) {
      if (error.message.includes('Failed to fetch')) {
        throw new Error('Cannot connect to the server. Please ensure the backend is running.');
      }
      throw error;
    }
  }

  /**
   * Verify OTP code submitted by the user
   * @param {string} email
   * @param {string} code - 6-digit OTP code
   * @returns {Promise<{success, token, user, message}>}
   */
  async verifyOTP(email, code) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/verify-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, code }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(this._parseDetail(data.detail, 'Verification failed'));
      }

      return data;
    } catch (error) {
      if (error.message.includes('Failed to fetch')) {
        throw new Error('Cannot connect to the server. Please ensure the backend is running.');
      }
      throw error;
    }
  }

  /**
   * Resend OTP to user's email
   * @param {string} email
   * @returns {Promise<{success, message}>}
   */
  async resendOTP(email) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/resend-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(this._parseDetail(data.detail, 'Resend failed'));
      }

      return data;
    } catch (error) {
      if (error.message.includes('Failed to fetch')) {
        throw new Error('Cannot connect to the server. Please ensure the backend is running.');
      }
      throw error;
    }
  }

  /**
   * Get current user info from a stored JWT token
   * @param {string} token - JWT token
   * @returns {Promise<UserResponse>}
   */
  async getCurrentUser(token) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
        headers: { 'Authorization': `Bearer ${token}` },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch user');
      }

      return data;
    } catch (error) {
      if (error.message.includes('Failed to fetch')) {
        throw new Error('Cannot connect to the server.');
      }
      throw error;
    }
  }
  /**
   * Save a prediction result to history
   * @param {string} token - JWT token
   * @param {Object} payload - { prediction_result, input_data, confidence_score, top_predictions, filename }
   * @returns {Promise<{success, id, message}>}
   */
  async saveHistory(token, payload) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to save history');
      }
      return data;
    } catch (error) {
      console.error('Error saving history:', error);
      throw error;
    }
  }

  /**
   * Get prediction history for the current user
   * @param {string} token - JWT token
   * @returns {Promise<{success, history, total}>}
   */
  async getHistory(token) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/history`, {
        headers: { 'Authorization': `Bearer ${token}` },
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to fetch history');
      }
      return data;
    } catch (error) {
      console.error('Error fetching history:', error);
      throw error;
    }
  }

  /**
   * Update an existing history record with learning roadmap / certification data
   * @param {string} token - JWT token
   * @param {number} historyId - history record ID returned by saveHistory
    * @param {Object} payload - { learning_roadmap_by_path?, certification_data_by_path?, learning_roadmap?, certification_data? }
   */
  async updateHistory(token, historyId, payload) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/history/${historyId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to update history');
      }
      return data;
    } catch (error) {
      console.error('Error updating history:', error);
      throw error;
    }
  }

  /**
   * Upload and attach the PDF resume to an existing history record.
   * @param {string} token - JWT token
   * @param {number} historyId - history record ID returned by saveHistory
   * @param {File} file - The PDF File object to store
   * @returns {Promise<{success, message}>}
   */
  async uploadHistoryResume(token, historyId, file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/history/${historyId}/resume`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to upload resume');
      }
      return data;
    } catch (error) {
      console.error('Error uploading history resume:', error);
      throw error;
    }
  }

  /**
   * Get the URL to download the stored resume PDF for a history record.
   * @param {number} historyId
   * @returns {string} URL string (use with Authorization header or open in fetch)
   */
  getHistoryResumeUrl(historyId) {
    return `${API_BASE_URL}/api/history/${historyId}/resume`;
  }

  // ─── Admin API ───────────────────────────────────────────────────────────────

  /**
   * Internal helper: admin authenticated fetch.
   * Throws on HTTP error, returns parsed JSON.
   */
  async _adminFetch(method, path, token, body = null) {
    const opts = {
      method,
      headers: {
        'Authorization': `Bearer ${token}`,
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    };
    const response = await fetch(`${API_BASE_URL}${path}`, opts);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(this._parseDetail(data.detail, data.message || 'Admin request failed'));
    }
    return data;
  }

  /**
   * Fetch admin dashboard statistics.
   * @param {string} token
   */
  async getAdminStats(token) {
    return this._adminFetch('GET', '/api/admin/stats', token);
  }

  /**
   * List users with optional search/filter/pagination.
   * @param {string} token
   * @param {Object} params - { search, verified, is_admin, page, page_size }
   */
  async getAdminUsers(token, params = {}) {
    const qs = new URLSearchParams(
      Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== ''))
    ).toString();
    return this._adminFetch('GET', `/api/admin/users${qs ? '?' + qs : ''}`, token);
  }

  /**
   * Create a new user from admin dashboard.
   * @param {string} token
   * @param {{ username, email, password, is_admin }} payload
   */
  async adminCreateUser(token, payload) {
    return this._adminFetch('POST', '/api/admin/users', token, payload);
  }

  /**
   * Update an existing user.
   * @param {string} token
   * @param {number} userId
   * @param {Object} payload - partial: { username?, email?, password?, is_admin?, is_verified? }
   */
  async adminUpdateUser(token, userId, payload) {
    return this._adminFetch('PUT', `/api/admin/users/${userId}`, token, payload);
  }

  /**
   * Delete a user by ID.
   * @param {string} token
   * @param {number} userId
   */
  async adminDeleteUser(token, userId) {
    return this._adminFetch('DELETE', `/api/admin/users/${userId}`, token);
  }

  /**
   * List all prediction history records (admin view).
   * @param {string} token
   * @param {Object} params - { search, user_id, page, page_size }
   */
  async getAdminHistory(token, params = {}) {
    const qs = new URLSearchParams(
      Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== '' && v !== 0))
    ).toString();
    return this._adminFetch('GET', `/api/admin/history${qs ? '?' + qs : ''}`, token);
  }

  /**
   * List users that have prediction history (admin view).
   * @param {string} token
   */
  async getAdminHistoryUsers(token) {
    return this._adminFetch('GET', '/api/admin/history/users', token);
  }

  /**
   * Delete a prediction history record by ID.
   * @param {string} token
   * @param {number} recordId
   */
  async adminDeleteHistory(token, recordId) {
    return this._adminFetch('DELETE', `/api/admin/history/${recordId}`, token);
  }

  /**
   * Fetch the stored resume PDF for a history record and return it as a Blob.
   * (Admin only — does not trigger a download)
   * @param {string} token
   * @param {number} recordId
   * @returns {Promise<Blob>}
   */
  async adminFetchResume(token, recordId) {
    const response = await fetch(`${API_BASE_URL}/api/admin/history/${recordId}/resume`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || 'Failed to load resume');
    }
    return response.blob();
  }

  /**
   * Download the stored resume PDF for a history record (admin only).
   * Returns a Blob that callers can use to trigger a file download.
   * @param {string} token
   * @param {number} recordId
   * @param {string} filename - suggested save name
   */
  async adminDownloadResume(token, recordId, filename) {
    const response = await fetch(`${API_BASE_URL}/api/admin/history/${recordId}/resume`, {
      headers: { 'Authorization': `Bearer ${token}` },
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || 'Failed to download resume');
    }
    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = filename || `resume_${recordId}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(blobUrl);
  }
}

// Export singleton instance
const apiService = new ApiService();
export default apiService;
