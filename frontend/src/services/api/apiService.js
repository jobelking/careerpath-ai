/**
 * API Service for CareerPath AI
 * Handles communication with the FastAPI backend
 */

// Use relative path for Docker compatibility
// When running in Docker, nginx will proxy /api to backend:8000
// When running locally, you can use http://localhost:8000
const API_BASE_URL = window.location.hostname === 'localhost' && (window.location.port === '5173' || window.location.port === '3000')
  ? 'http://localhost:8000'  // Development mode (Vite)
  : '';  // Production mode (Docker/nginx proxy)

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
}

// Export singleton instance
const apiService = new ApiService();
export default apiService;
