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
  const [showAllPaths, setShowAllPaths] = useState(false);
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

  // --------------------
  // Confidence (31 classes) - Non-technical UI text
  // --------------------

  const NUM_CLASSES = 31;

  const toNum = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  };

  /**
   * IMPORTANT:
   * - Confidence LABEL is computed using RAW probabilities (across all 31 careers).
   * - NO normalization allowed.
   */
  const calculateConfidenceLevel = (predictions, numClasses = NUM_CLASSES) => {
    if (!predictions || predictions.length === 0) return "Exploratory";
    if (predictions.length === 1) return "Shared Fit";

    const chance = 100 / numClasses; // ~3.23% for 31

    // Prefer raw_confidence; fallback to confidence if needed
    const p1 = toNum(predictions[0].raw_confidence ?? predictions[0].confidence);
    const p2 = toNum(predictions[1].raw_confidence ?? predictions[1].confidence);
    const p3 = toNum(predictions[2]?.raw_confidence ?? predictions[2]?.confidence);

    const margin = p1 - p2;
    const mass3 = p1 + p2 + p3;

    // Clear Match: very strong top match OR clearly stands out
    if (p1 >= 6 * chance) return "Clear Match";                 // ~19%+
    if (margin >= 8 && p1 >= 4 * chance) return "Clear Match";  // gap 8%+ and top1 ~13%+

    // Shared Fit: strong top match but close alternatives OR focused in top-3
    if (p1 >= 3 * chance && margin >= 3) return "Shared Fit";    // top1 ~9.7%+ and gap 3%+
    if (mass3 >= 28 && p1 >= 3 * chance) return "Shared Fit";    // top-3 raw ~28%+ indicates focus

    // Exploratory: no single role clearly stands out
    return "Exploratory";
  };

  /**
   * User-facing explanation (non-technical, honest, and consistent with the logic).
   */
  const getConfidenceExplanation = (predictions, numClasses = NUM_CLASSES) => {
    if (!predictions || predictions.length === 0) {
      return "This result is based on how closely your background matches different career paths.";
    }

    const level = calculateConfidenceLevel(predictions, numClasses);

    const topCareer = predictions[0]?.career_path || "this role";

    if (level === "Clear Match") {
      return `Clear Match: Your background strongly aligns with "${topCareer}". This path stands out as a primary fit based on your current skills.`;
    }

    if (level === "Shared Fit") {
      return `Shared Fit: You show strong potential in "${topCareer}" and other related fields. Your skills overlap across multiple career paths.`;
    }

    return `Exploratory: Your skills apply to many different areas without a single dominant match. Use these results as a starting point for exploration.`;
  };


  return (
    <div className="dashboard-container">
      {/* Header/Navigation */}
      <header className="dashboard-header">
        <div className="header-content">
          <h1 className="dashboard-brand" onClick={() => navigate('/')}>
            <Logo variant="modern" />
          </h1>

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
                      <div
                        className={`confidence-badge ${calculateConfidenceLevel(predictionResults.top_predictions).toLowerCase().replace(' ', '-')}`}
                        data-tooltip={getConfidenceExplanation(predictionResults.top_predictions)}
                      >
                        <span className="confidence-icon">i</span>
                        {calculateConfidenceLevel(predictionResults.top_predictions)}
                      </div>
                    </div>
                    <p className="results-context">
                      These results are based on an analysis of your skills and experience. The percentages shown are raw probabilities across 31 potential career paths.
                    </p>
                  </div>

                  {predictionResults && predictionResults.top_predictions && predictionResults.top_predictions.length > 0 && (
                    <>
                      {/* 1. Summary Section (Primary Match) */}
                      <div className="primary-match-section">
                        <h4 className="section-label">Summary: Primary Match</h4>
                        <div className="primary-prediction-card">
                          <div className="primary-card-header">
                            <div className="primary-icon-wrapper">
                              {getCareerIcon(predictionResults.top_predictions[0].career_path)}
                            </div>
                            <div className="primary-info">
                              <h3 className="primary-career-name">
                                {predictionResults.top_predictions[0].career_path}
                              </h3>
                              <span className="primary-label">Top Ranked Result</span>
                            </div>
                          </div>

                          <div className="primary-match-bar">
                            <div className="match-bar-bg">
                              <div
                                className="match-bar-fill primary"
                                style={{ width: `${predictionResults.top_predictions[0].raw_confidence || predictionResults.top_predictions[0].confidence}%` }}
                              ></div>
                            </div>
                            <span className="match-percentage">
                              {(predictionResults.top_predictions[0].raw_confidence || predictionResults.top_predictions[0].confidence).toFixed(1)}% Raw Confidence
                            </span>
                          </div>
                        </div>
                      </div>




                      {/* 2. Next Best Matches (Rank 2 & 3) */}
                      {predictionResults.top_predictions.length > 1 && (
                        <div className="secondary-matches-section">
                          <h4 className="section-label">Next Best Matches</h4>
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
                                    </div>
                                  </div>

                                  <div className="secondary-right">
                                    <div className="secondary-bar-container">
                                      <div
                                        className="secondary-bar-fill"
                                        style={{ width: `${career.raw_confidence || career.confidence}%` }}
                                      ></div>
                                    </div>
                                    <span className="secondary-percentage">
                                      {(career.raw_confidence || career.confidence).toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* 3. Full Transparency Section */}
                      <div className="transparency-section">
                        <div className="transparency-header">
                          <button
                            className="expand-button"
                            onClick={() => setShowAllPaths(!showAllPaths)}
                          >
                            {showAllPaths ? "Hide All Career Paths" : "View All Career Paths (31)"}
                          </button>
                        </div>

                        {showAllPaths && (
                          <div className="all-paths-grid">
                            <div className="paths-header-row">
                              <span>Rank</span>
                              <span>Career Path</span>
                              <span>Confidence</span>
                            </div>
                            {predictionResults.top_predictions.map((career, index) => (
                              <div key={index} className="path-item">
                                <span className="path-rank">#{index + 1}</span>
                                <span className="path-name">{career.career_path}</span>
                                <span className="path-percent">
                                  {(career.raw_confidence || career.confidence).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
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