import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../../components/common/Logo';
import apiService from '../../services/api/apiService';
import { careerIcons } from '../../utils/careerIcons';
import { otherIcons } from '../../utils/otherIcons';
import { useDashboard } from '../../context/DashboardContext';
import { useAuth } from '../../context/AuthContext';
import CareerPathsModal from '../../components/common/CareerPathsModal/CareerPathsModal';
import './Dashboard.css';

const Dashboard = () => {
  const navigate = useNavigate();
  const { currentUser, logout, getToken } = useAuth();
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [showCareerPaths, setShowCareerPaths] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  // Use context for persistent state
  const {
    predictionResults,
    setPredictionResults,
    uploadedFileName,
    setUploadedFileName,
    uploadedFile,
    setUploadedFile,
    setResumeText,
    clearResults,
    setHistoryRecordId,
  } = useDashboard();

  // Local state (doesn't need persistence)
  const [isLoading, setIsLoading] = useState(false);
  const [showAllPaths, setShowAllPaths] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Derive showResults from predictionResults
  const showResults = predictionResults !== null;

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
      setUploadedFileName(file.name);
      setError(null);
      // Don't clear results when new file is selected, only when analyzed
    }
  };



  const [previewUrl, setPreviewUrl] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  // Manage object URL lifecycle
  useEffect(() => {
    let objectUrl = null;
    if (uploadedFile) {
      objectUrl = URL.createObjectURL(uploadedFile);
      setPreviewUrl(objectUrl);
    } else {
      setPreviewUrl(null);
    }
    // Reset preview state when file changes
    setShowPreview(false);
    setIsClosing(false);

    return () => {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [uploadedFile]);

  const handleClosePreview = () => {
    setIsClosing(true);
    setTimeout(() => {
      setShowPreview(false);
      setIsClosing(false);
    }, 150);
  };

  const handleTogglePreview = () => {
    if (showPreview) {
      handleClosePreview();
    } else if (previewUrl) {
      setShowPreview(true);
    }
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) {
      setError('Please upload a resume first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setShowPreview(false);

    try {
      // Call API to predict career path
      const result = await apiService.predictCareerPath(uploadedFile);

      if (result.success) {
        setPredictionResults(result);
        setUploadedFileName(uploadedFile.name);
        // Save resume text for learning roadmap
        if (result.resume_text) {
          setResumeText(result.resume_text);
        }

        // ── Auto-save prediction to history (fire-and-forget) ──────────────
        const token = getToken();
        if (token) {
          const fileSnapshot = uploadedFile; // capture before any state reset
          apiService.saveHistory(token, {
            prediction_result: result.prediction,
            confidence_score: result.raw_confidence,
            top_predictions: result.top_predictions?.slice(0, 3) ?? [],
            filename: uploadedFile.name,
            input_data: result.resume_text
              ? result.resume_text.slice(0, 500)
              : null,
          }).then((saved) => {
            // Store the record ID so LearnMore can PATCH roadmap/certs onto it
            if (saved?.id) {
              setHistoryRecordId(saved.id);
              // Upload the PDF so it's permanently linked to this history record
              apiService.uploadHistoryResume(token, saved.id, fileSnapshot).catch((err) => {
                console.warn('Resume PDF upload failed (non-critical):', err);
              });
            }
          }).catch((err) => {
            console.warn('History save failed (non-critical):', err);
          });
        }
      } else {
        setError('Failed to analyze resume. Please try again.');
        // Clear file so user can select a new one
        setUploadedFile(null);
        setUploadedFileName('');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message || 'An error occurred while analyzing your resume. Please ensure the backend server is running.');
      // Clear file so user can select a new one
      setUploadedFile(null);
      setUploadedFileName('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setUploadedFile(null);
    clearResults();
    setShowAllPaths(false);
    setError(null);
    setPreviewUrl(null);
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
  // Confidence (26 classes) - Non-technical UI text
  // --------------------

  const NUM_CLASSES = 26;

  const toNum = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  };

  /**
   * Calculates a User-Friendly "Profile Fit" score from the Raw Probability.
   * Uses historical accuracy from Feb 2026 calibration data.
   * Based on calibration results (26 classes, 1583 test samples, 81.87% overall accuracy):
   *   0-5% raw → N/A (only 2 samples, statistically unreliable - use linear interpolation)
   *   5-10% raw → 45% accuracy (199 samples)
   *   10-15% raw → 65% accuracy (184 samples)
   *   15-20% raw → 83% accuracy (145 samples)
   *   20-30% raw → 85% accuracy (198 samples)
   *   30-50% raw → 90% accuracy (276 samples)
   *   50-100% raw → 95% accuracy (579 samples)
   * 
   *   0-5% raw → ramps 0→35% (linear, connects smoothly to the 5-10% bin)
   */
  const calculateProfileFit = (rawProbability) => {
    const p = rawProbability;

    if (p < 5) return Math.round((p / 5) * 35);              // 0-5% → 0-35% (linear ramp into the 5-10% bin start)
    if (p < 10) return Math.round(35 + ((p - 5) / 5) * 15);  // 5-10% → 35-50% (centered at 45%, interpolated for UI)
    if (p < 15) return Math.round(50 + ((p - 10) / 5) * 15); // 10-15% → 50-65%
    if (p < 20) return Math.round(65 + ((p - 15) / 5) * 18); // 15-20% → 65-83%
    if (p < 30) return Math.round(83 + ((p - 20) / 10) * 2); // 20-30% → 83-85%
    if (p < 50) return Math.round(85 + ((p - 30) / 20) * 5); // 30-50% → 85-90%
    return Math.round(90 + ((p - 50) / 50) * 5);             // 50-100% → 90-95%
  };

  return (
    <div className="dashboard-container">
      {/* Header/Navigation */}
      <header className="dashboard-header">
        <div className="header-content">
          <h1 className="dashboard-brand" onClick={() => navigate('/')}>
            <Logo variant="modern" />
          </h1>

          {/* Hamburger Button (mobile only) */}
          <button
            className={`hamburger-btn ${menuOpen ? 'open' : ''}`}
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle menu"
          >
            <span></span>
            <span></span>
            <span></span>
          </button>

          {/* Desktop nav */}
          <div className="dashboard-header-actions">
            {currentUser && (
              <span className="dashboard-greeting">Hello, {currentUser.username}</span>
            )}
            <button
              className="dashboard-history-btn"
              onClick={() => setShowCareerPaths(true)}
              id="career-paths-btn"
            >
              Career Paths
            </button>
            <button
              className="dashboard-history-btn"
              onClick={() => navigate('/history')}
            >
              History
            </button>
            {/* Admin Panel button — only visible to admin users */}
            {currentUser?.is_admin && (
              <button
                className="dashboard-admin-btn"
                onClick={() => navigate('/admin')}
              >
                🛡 Admin Panel
              </button>
            )}
            <button
              className="dashboard-logout-btn"
              onClick={() => setShowLogoutConfirm(true)}
            >
              Logout
            </button>
          </div>
        </div>

        {/* Mobile nav drawer */}
        {menuOpen && (
          <div className="mobile-nav-drawer">
            {currentUser && (
              <span className="mobile-nav-greeting">Hello, {currentUser.username}</span>
            )}
            <button
              className="dashboard-history-btn mobile-nav-btn"
              onClick={() => { setShowCareerPaths(true); setMenuOpen(false); }}
            >
              Career Paths
            </button>
            <button
              className="dashboard-history-btn mobile-nav-btn"
              onClick={() => { navigate('/history'); setMenuOpen(false); }}
            >
              History
            </button>
            {/* Admin Panel button in mobile drawer — only for admins */}
            {currentUser?.is_admin && (
              <button
                className="dashboard-admin-btn mobile-nav-btn"
                onClick={() => { navigate('/admin'); setMenuOpen(false); }}
              >
                🛡 Admin Panel
              </button>
            )}
            <button
              className="dashboard-logout-btn mobile-nav-btn"
              onClick={() => { setShowLogoutConfirm(true); setMenuOpen(false); }}
            >
              Logout
            </button>
          </div>
        )}

        {/* Logout confirmation banner */}
        {showLogoutConfirm && (
          <div className="logout-confirm-bar">
            <span>Are you sure you want to logout?</span>
            <div className="logout-confirm-actions">
              <button className="logout-confirm-yes" onClick={handleLogout}>Yes, Logout</button>
              <button className="logout-confirm-cancel" onClick={() => setShowLogoutConfirm(false)}>Cancel</button>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="dashboard-main">
        <div className="dashboard-content">
          {/* Welcome Section */}


          {/* Layout Area */}
          <div className={`dashboard-grid ${!showResults && !isLoading && !showPreview ? 'centered' : ''} ${showAllPaths ? 'paths-expanded' : ''}`}>
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
                  <p>Drag and drop your resume below</p>
                </div>

                <div
                  className="upload-area"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                >
                  <input
                    type="file"
                    id="resume-upload"
                    ref={fileInputRef}
                    className="file-input"
                    accept=".pdf"
                    onChange={handleFileUpload}
                  />
                  <div className="upload-label">
                    {uploadedFile || (showResults && uploadedFileName) ? (
                      <div className="file-uploaded">
                        <span></span>
                        <span className="file-icon">{React.createElement(otherIcons["FaCheck"])}</span>
                        <span className="file-name">{uploadedFile ? uploadedFile.name : uploadedFileName}</span>
                        {!showResults && uploadedFile && (
                          <button
                            className="change-file-btn"
                            onClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setUploadedFile(null);
                              setUploadedFileName('');
                              setError(null);
                              if (fileInputRef.current) fileInputRef.current.value = '';
                            }}
                          >
                            Change File
                          </button>
                        )}
                        <span></span>
                      </div>
                    ) : (
                      <div className="upload-prompt">
                        <div className="upload-cloud">
                          {React.createElement(otherIcons["FaUpload"], { color: "#2563eb" })}
                        </div>
                        <span className="upload-text">Drag and drop your resume</span>
                        <span className="upload-formats">PDF only (Max 10MB)</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="upload-actions">
                  {/* Upload Button */}
                  <button
                    className={`upload-button ${showResults ? 'reset-mode' : ''}`}
                    onClick={() => {
                      if (showResults) {
                        handleReset();
                      } else if (!uploadedFile) {
                        fileInputRef.current?.click();
                      } else {
                        handleAnalyze();
                      }
                    }}
                    disabled={isLoading}
                  >
                    {isLoading ? 'Analyzing...' : showResults ? 'Add New Resume' : uploadedFile ? 'Upload and Analyze' : 'Select Resume'}
                  </button>

                  {/* View Resume Button (Secondary) */}
                  <button
                    className="preview-button"
                    onClick={handleTogglePreview}
                    disabled={isLoading || !uploadedFile}
                  >
                    {showPreview ? 'Close Resume PDF' : 'View Resume PDF'}
                  </button>
                </div>

                {/* Error Message */}
                {error && (
                  <div className="error-message">
                    <span className="error-icon" style={{ display: 'flex', alignItems: 'center', marginTop: '2px' }}>
                      {React.createElement(otherIcons["FaExclamationTriangle"], { color: '#ef4444' })}
                    </span>
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

            </div>

            {/* Right Column - Results Section */}
            <div className="right-column">
              {isLoading ? (
                <div className="loading-section">
                  <div className="loading-spinner"></div>
                  <h3>Analyzing Your Resume...</h3>
                  <p>Please wait while our AI processes your information</p>
                </div>
              ) : showPreview ? (
                // Resume Preview Mode
                <div className={`results-section preview-mode ${isClosing ? 'closing' : ''}`}>
                  <div className="results-header">
                    <div className="results-title-row">
                      <h3>Resume Preview</h3>
                      <button className="close-preview-btn" onClick={handleClosePreview} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        {React.createElement(otherIcons["FaTimes"])} Close
                      </button>
                    </div>
                  </div>
                  <div className="preview-content-full">
                    <iframe
                      src={previewUrl}
                      className="preview-iframe-full"
                      title="Resume Preview"
                    />
                  </div>
                </div>
              ) : showResults ? (
                <div className="results-section">
                  <div className="results-header">
                    <div className="results-title-row">
                      <h3>Your Career Analysis</h3>
                    </div>
                    <p className="results-context">
                      These results compare your profile against 26 career paths. The "Profile Fit" score reflects alignment strength, not probability of success.
                    </p>

                  </div>

                  {predictionResults && predictionResults.top_predictions && predictionResults.top_predictions.length > 0 && (
                    <>
                      {/* 1. Summary Section (Strongest Match) */}
                      {(() => {
                        const reliability = calculateProfileFit(predictionResults.top_predictions[0].raw_confidence);
                        return (
                          <div className="primary-match-section">
                            <h4 className="section-label">Strongest Match</h4>
                            <div className="primary-prediction-card">
                              <div className="primary-card-header">
                                <div className="primary-icon-wrapper">
                                  {getCareerIcon(predictionResults.top_predictions[0].career_path)}
                                </div>
                                <div className="primary-info">
                                  <h3 className="primary-career-name">
                                    {predictionResults.top_predictions[0].career_path}
                                  </h3>
                                  <span className="primary-label">Rank #1 based on your profile</span>
                                </div>
                              </div>

                              <div className="primary-profile-fit">
                                <div className="profile-fit-header">
                                  <span className="profile-fit-label">Profile Fit</span>
                                  <span className="profile-fit-value">
                                    {reliability}%
                                    <span style={{ fontSize: '0.75rem', color: '#9ca3af', marginLeft: '6px', fontWeight: 'normal' }}>
                                      ({predictionResults.top_predictions[0].raw_confidence.toFixed(1)}% raw)
                                    </span>
                                  </span>
                                </div>
                                <div className="profile-fit-bar">
                                  <div className="profile-fit-bar-fill" style={{ width: `${reliability}%` }}></div>
                                </div>
                              </div>

                              <button
                                className="learn-more-button"
                                onClick={() => navigate('/learnmore')}
                              >
                                <span>View Detailed Analysis</span>
                                {React.createElement(otherIcons["FaArrowRight"], { size: 14 })}
                              </button>
                            </div>
                          </div>
                        );
                      })()}




                      {/* 2. Next Best Matches (Rank 2 & 3) */}
                      {predictionResults.top_predictions.length > 1 && (() => {
                        return (
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
                                      </div>
                                    </div>

                                    <div className="secondary-profile-fit">
                                      <div className="secondary-fit-header">
                                        <span className="secondary-fit-label">Profile Fit</span>
                                        <span className="secondary-fit-value">
                                          {calculateProfileFit(career.raw_confidence)}%
                                          <span style={{ fontSize: '0.7rem', color: '#9ca3af', marginLeft: '4px', fontWeight: 'normal' }}>
                                            ({career.raw_confidence.toFixed(1)}% raw)
                                          </span>
                                        </span>
                                      </div>
                                      <div className="secondary-fit-bar">
                                        <div className="secondary-fit-bar-fill" style={{ width: `${calculateProfileFit(career.raw_confidence)}%` }}></div>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })()}

                      {/* 3. Full Transparency Section */}
                      <div className="transparency-section">
                        <div className="transparency-header">
                          <button
                            className="expand-button"
                            onClick={() => setShowAllPaths(!showAllPaths)}
                          >
                            {showAllPaths ? "Hide All Career Paths" : "View All Career Paths"}
                          </button>
                        </div>

                        {showAllPaths && (
                          <div className="all-paths-grid">
                            <div className="paths-header-row">
                              <span>Rank</span>
                              <span>Career Path</span>
                            </div>
                            {predictionResults.top_predictions.map((career, index) => (
                              <div key={index} className="path-item">
                                <span className="path-rank">#{index + 1}</span>
                                <span className="path-name">{career.career_path}</span>
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

      {/* Career Paths Modal */}
      {showCareerPaths && (
        <CareerPathsModal onClose={() => setShowCareerPaths(false)} />
      )}
    </div>
  );
};

export default Dashboard;
