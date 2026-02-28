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
}

// Export singleton instance
const apiService = new ApiService();
export default apiService;
