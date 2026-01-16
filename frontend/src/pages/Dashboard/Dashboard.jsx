import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../../components/common/Logo';
import apiService from '../../services/api/apiService';
import { careerIcons } from '../../utils/careerIcons';
import { otherIcons } from '../../utils/otherIcons';
import './Dashboard.css';

const Dashboard = () => {
  const navigate = useNavigate();
  const [uploadedFile, setUploadedFile] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResults, setPredictionResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        setError('Only PDF files are supported. Please upload a PDF resume.');
        return;
      }

      // Validate file size (10MB max)
      const maxSize = 10 * 1024 * 1024; // 10MB
      if (file.size > maxSize) {
        setError(`File size exceeds 10MB. Your file is ${(file.size / (1024 * 1024)).toFixed(2)}MB`);
        return;
      }

      setUploadedFile(file);
      setError(null);
      setShowResults(false);
    }
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) {
      setError('Please upload a resume first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setShowResults(false);

    try {
      // Call API to predict career path
      const result = await apiService.predictCareerPath(uploadedFile);

      if (result.success) {
        setPredictionResults(result);
        setShowResults(true);
      } else {
        setError('Failed to analyze resume. Please try again.');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message || 'An error occurred while analyzing your resume. Please ensure the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      // Validate file type
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        setError('Only PDF files are supported. Please upload a PDF resume.');
        return;
      }

      // Validate file size
      const maxSize = 10 * 1024 * 1024; // 10MB
      if (file.size > maxSize) {
        setError(`File size exceeds 10MB. Your file is ${(file.size / (1024 * 1024)).toFixed(2)}MB`);
        return;
      }

      setUploadedFile(file);
      setError(null);
      setShowResults(false);
    }
  };

  // Get career icon based on career name using react-icons mapping
  const getCareerIcon = (careerName) => {
    const Icon = careerIcons[careerName] || careerIcons["Software Engineer"]; // fallback icon
    return <Icon size={32} color="#2563eb" />;
  };

  // Calculate confidence level based on probability distribution
  const calculateConfidenceLevel = (predictions) => {
    if (!predictions || predictions.length < 2) return "High Confidence";

    const topProb = predictions[0].confidence;
    const secondProb = predictions[1].confidence;
    const gap = topProb - secondProb;

    // High confidence: large gap between top 2
    if (gap > 15) return "High Confidence";
    // Mixed profile: moderate gap, distributed probabilities
    if (gap > 8) return "Mixed Profile";
    // Exploratory: small gaps, very distributed
    return "Exploratory Match";
  };

  return (
    <div className="dashboard-container">
      {/* Header/Navigation */}
      <header className="dashboard-header">
        <div className="header-content">
          <h1 className="dashboard-brand" onClick={() => navigate('/')}>
            <Logo variant="modern" />
          </h1>
          <nav className="dashboard-nav">
            <button className="nav-link active">Dashboard</button>
            <button className="nav-link" onClick={() => navigate('/history')}>History</button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="dashboard-main">
        <div className="dashboard-content">
          {/* Welcome Section */}


          {/* Two-Column Layout */}
          <div className="dashboard-grid">
            {/* Left Column - Upload Section */}
            <div className="left-column">
              <div className="upload-card">
                <div className="upload-header">
                  <div className="upload-icon-wrapper">
                    <span className="upload-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="#2563eb">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm4 18H6V4h7v5h5v11z" />
                      </svg>
                    </span>
                  </div>
                  <h3>Upload Your Resume</h3>
                  <p>Drag and drop your resume or click to browse</p>
                </div>

                <div
                  className="upload-area"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                >
                  <input
                    type="file"
                    id="resume-upload"
                    className="file-input"
                    accept=".pdf,.doc,.docx"
                    onChange={handleFileUpload}
                  />
                  <label htmlFor="resume-upload" className="upload-label">
                    {uploadedFile ? (
                      <div className="file-uploaded">
                        <span></span>
                        <span className="file-icon">✓</span>
                        <span className="file-name">{uploadedFile.name}</span>
                        <span></span>
                      </div>
                    ) : (
                      <div className="upload-prompt">
                        <div className="upload-cloud">
                          {React.createElement(otherIcons["FaUpload"], { color: "#2563eb" })}
                        </div>
                        <span className="upload-text">Click to upload or drag and drop</span>
                        <span className="upload-formats">PDF, DOC, DOCX (Max 10MB)</span>
                      </div>
                    )}
                  </label>
                </div>

                {/* Upload Button */}
                <button
                  className="upload-button"
                  onClick={handleAnalyze}
                  disabled={!uploadedFile || isLoading}
                >
                  {isLoading ? 'Analyzing...' : 'Upload and Analyze'}
                </button>

                {/* Error Message */}
                {error && (
                  <div className="error-message">
                    <span className="error-icon">⚠️</span>
                    <span>{error}</span>
                  </div>
                )}

                {/* Privacy Note */}
                <div className="privacy-note">
                  <span className="info-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="#2563eb">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
                    </svg>
                  </span>
                  <p>Please remove sensitive information (e.g., address, SSN, personal contact details) before uploading your resume.</p>
                </div>
              </div>

              {/* Analysis Info Card */}
              <div className="info-card">
                <h4>How It Works</h4>
                <ul className="info-list">
                  <li>
                    <span className="step-number">1</span>
                    <span>Upload your resume in PDF format</span>
                  </li>
                  <li>
                    <span className="step-number">2</span>
                    <span>Our AI analyzes your skills and experience</span>
                  </li>
                  <li>
                    <span className="step-number">3</span>
                    <span>Get personalized career path recommendations</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Right Column - Results Section */}
            <div className="right-column">
              {isLoading ? (
                <div className="loading-section">
                  <div className="loading-spinner"></div>
                  <h3>Analyzing Your Resume...</h3>
                  <p>Please wait while our AI processes your information</p>
                </div>
              ) : showResults ? (
                <div className="results-section">
                  <div className="results-header">
                    <div className="results-title-row">
                      <h3>Your Career Analysis</h3>
                      <div className={`confidence-badge ${calculateConfidenceLevel(predictionResults.top_predictions).toLowerCase().replace(' ', '-')}`}>
                        <span className="confidence-icon">●</span>
                        {calculateConfidenceLevel(predictionResults.top_predictions)}
                      </div>
                    </div>
                    <p className="results-context">
                      Your resume shows strengths across multiple career areas. The primary match reflects
                      your strongest overall alignment, while others indicate overlapping skill sets.
                    </p>
                  </div>

                  {predictionResults && predictionResults.top_predictions && predictionResults.top_predictions.length > 0 && (
                    <>
                      {/* Primary Career Match */}
                      <div className="primary-match-section">
                        <h4 className="section-label">Primary Career Match</h4>
                        <div className="primary-prediction-card">
                          <div className="primary-card-header">
                            <div className="primary-icon-wrapper">
                              {getCareerIcon(predictionResults.top_predictions[0].career_path)}
                            </div>
                            <div className="primary-info">
                              <h3 className="primary-career-name">
                                {predictionResults.top_predictions[0].career_path}
                              </h3>
                              <span className="primary-label">Strongest Overall Alignment</span>
                            </div>
                          </div>

                          <div className="primary-match-bar">
                            <div className="match-bar-bg">
                              <div
                                className="match-bar-fill primary"
                                style={{ width: `${predictionResults.top_predictions[0].confidence}%` }}
                              ></div>
                            </div>
                            <span className="match-percentage">
                              {Math.round(predictionResults.top_predictions[0].confidence)}% Relative Match
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Secondary Matches */}
                      {predictionResults.top_predictions.length > 1 && (
                        <div className="secondary-matches-section">
                          <h4 className="section-label">Also Strong Matches</h4>
                          <div className="secondary-predictions-list">
                            {predictionResults.top_predictions.slice(1, 3).map((career, index) => (
                              <div key={index + 1} className="secondary-prediction-card">
                                <div className="secondary-card-content">
                                  <div className="secondary-left">
                                    <div className="secondary-rank">#{index + 2}</div>
                                    <div className="secondary-icon">
                                      {getCareerIcon(career.career_path)}
                                    </div>
                                    <div className="secondary-info">
                                      <h5 className="secondary-career-name">{career.career_path}</h5>
                                      <span className="secondary-label">Alternative Path</span>
                                    </div>
                                  </div>

                                  <div className="secondary-right">
                                    <div className="secondary-bar-container">
                                      <div
                                        className="secondary-bar-fill"
                                        style={{ width: `${career.confidence}%` }}
                                        title={`${Math.round(career.confidence)}% match`}
                                      ></div>
                                    </div>
                                    <span className="secondary-percentage">
                                      {Math.round(career.confidence)}%
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              ) : (
                <div className="placeholder-section">
                  <div className="placeholder-icon">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="#2563eb">
                      <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z" />
                    </svg>
                  </div>
                  <h3>Your Career Predictions Will Appear Here</h3>
                  <p>Upload your resume to see AI-powered career recommendations tailored to your profile</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;