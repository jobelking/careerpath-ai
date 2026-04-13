import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../../components/common/Logo';
import apiService from '../../services/api/apiService';
import { careerIcons } from '../../utils/careerIcons';
import { otherIcons } from '../../utils/otherIcons';
import { authIcons } from '../../utils/authIcons';
import { useDashboard } from '../../context/DashboardContext';
import { useAuth } from '../../context/AuthContext';
import CareerPathsModal from '../../components/common/CareerPathsModal/CareerPathsModal';
import ChangePasswordModal from '../../components/auth/ChangePasswordModal';
import './Dashboard.css';

const Dashboard = () => {
  const navigate = useNavigate();
  const { currentUser, logout, getToken } = useAuth();
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [showCareerPaths, setShowCareerPaths] = useState(false);
  const [showChangePassword, setShowChangePassword] = useState(false);
  const [userDropdownOpen, setUserDropdownOpen] = useState(false);
  const userDropdownRef = useRef(null);

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

  // Close user dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (userDropdownRef.current && !userDropdownRef.current.contains(e.target)) {
        setUserDropdownOpen(false);
      }
    };
    if (userDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [userDropdownOpen]);

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
            // Strip NUL bytes (0x00) — PostgreSQL rejects them in string literals
            input_data: result.resume_text
              ? result.resume_text.replace(/\0/g, '').slice(0, 500)
              : null,
            extracted_keywords: result.extracted_keywords ?? [],
            extracted_keywords_by_path: result.extracted_keywords_by_path ?? {},
            total_distinctive_keywords: result.total_distinctive_keywords ?? 0,
            total_distinctive_keywords_by_path: result.total_distinctive_keywords_by_path ?? {},
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

  const handleViewDetailedAnalysis = (careerPath) => {
    if (!careerPath) return;
    navigate(`/learnmore?career=${encodeURIComponent(careerPath)}`);
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
   * Uses historical accuracy from Apr 2026 calibration data.
   * Based on calibration results (26 classes, 1497 test samples, 84.97% overall accuracy):
   *   0-5% raw → 0% accuracy (2 samples, statistically unreliable - use linear interpolation)
   *   5-10% raw → 49% accuracy (146 samples)
   *   10-15% raw → 62% accuracy (145 samples)
   *   15-20% raw → 80% accuracy (128 samples)
   *   20-30% raw → 83% accuracy (191 samples)
   *   30-50% raw → 92% accuracy (237 samples)
   *   50-100% raw → 97% accuracy (648 samples)
   * 
   *   0-5% raw → ramps 0→35% (linear, connects smoothly to the 5-10% bin)
   */
  const calculateProfileFit = (rawProbability) => {
    const p = rawProbability;

    if (p < 5) return Math.round((p / 5) * 35);              // 0-5%   → 0-35%  (linear ramp into the 5-10% bin start)
    if (p < 10) return Math.round(35 + ((p - 5) / 5) * 14);  // 5-10%  → 35-49% (centered at ~49%, per calibration)
    if (p < 15) return Math.round(49 + ((p - 10) / 5) * 13); // 10-15% → 49-62%
    if (p < 20) return Math.round(62 + ((p - 15) / 5) * 18); // 15-20% → 62-80%
    if (p < 30) return Math.round(80 + ((p - 20) / 10) * 3); // 20-30% → 80-83%
    if (p < 50) return Math.round(83 + ((p - 30) / 20) * 9); // 30-50% → 83-92%
    return Math.round(92 + ((p - 50) / 50) * 5);             // 50-100% → 92-97%
  };

  // Determine current dashboard mode for layout
  const hasFile = uploadedFile || (showResults && uploadedFileName);
  const isCompactUpload = showResults || isLoading || showPreview;

  return (
    <div className="dashboard-container">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="dashboard-header">
        <div className="header-content">
          <div className="dashboard-header-left">
            <h1 className="dashboard-brand" onClick={() => navigate('/')}>
              <Logo variant="modern" />
            </h1>
          </div>

          <nav className="dashboard-top-nav">
            <button className="dashboard-nav-tab active" type="button" disabled>Dashboard</button>
            <button className="dashboard-nav-tab" onClick={() => setShowCareerPaths(true)}>Career Paths</button>
            <button className="dashboard-nav-tab" onClick={() => navigate('/history')}>History</button>
          </nav>

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

          {/* Desktop actions */}
          <div className="dashboard-header-right">
            {/* Admin Panel button — only visible to admin users */}
            {currentUser?.is_admin && (
              <button
                className="dashboard-admin-btn"
                onClick={() => navigate('/admin')}
              >
                🛡 Admin
              </button>
            )}

            {/* User dropdown (profile chip → clickable) */}
            {currentUser && (
              <div className="user-dropdown-container" ref={userDropdownRef}>
                <button
                  className={`user-dropdown-trigger ${userDropdownOpen ? 'open' : ''}`}
                  onClick={() => setUserDropdownOpen(v => !v)}
                  aria-expanded={userDropdownOpen}
                  aria-haspopup="true"
                >
                  <span className="dashboard-profile-dot" aria-hidden="true"></span>
                  <span className="dashboard-greeting">{currentUser.username}</span>
                  {React.createElement(authIcons['FaChevronDown'], {
                    className: `user-dropdown-chevron ${userDropdownOpen ? 'rotated' : ''}`,
                    size: 11
                  })}
                </button>

                {userDropdownOpen && (
                  <div className="user-dropdown-menu">
                    <div className="user-dropdown-header">
                      <span className="user-dropdown-email">{currentUser.email}</span>
                    </div>
                    <button
                      className="user-dropdown-item"
                      onClick={() => {
                        setUserDropdownOpen(false);
                        setShowChangePassword(true);
                      }}
                    >
                      {React.createElement(authIcons['FaKey'], { size: 14 })}
                      Change Password
                    </button>
                    <div className="user-dropdown-divider"></div>
                    <button
                      className="user-dropdown-item user-dropdown-item--danger"
                      onClick={() => {
                        setUserDropdownOpen(false);
                        setShowLogoutConfirm(true);
                      }}
                    >
                      {React.createElement(authIcons['FaSignOutAlt'], { size: 14 })}
                      Logout
                    </button>
                  </div>
                )}
              </div>
            )}
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
              type="button"
              disabled
            >
              Dashboard
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
            <div className="mobile-nav-divider"></div>
            <button
              className="mobile-nav-btn mobile-nav-btn--subtle"
              onClick={() => { setShowChangePassword(true); setMenuOpen(false); }}
            >
              Change Password
            </button>
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

      {/* ── Main Content ────────────────────────────────────────── */}
      <main className="dashboard-main">
        <div className="dashboard-flow">

          {/* ── Upload Section ─────────────────────────────────── */}
          <section className={`upload-section ${isCompactUpload ? 'compact' : 'hero'}`}>
            <div className="upload-card">

              {/* Compact mode: horizontal inline strip */}
              {isCompactUpload ? (
                <div className="upload-compact-row">
                  <div className="upload-compact-file">
                    <span className="upload-compact-icon">
                      {React.createElement(otherIcons["FaCheck"])}
                    </span>
                    <span className="upload-compact-name">{uploadedFile ? uploadedFile.name : uploadedFileName}</span>
                  </div>
                  <div className="upload-compact-actions">
                    <button
                      className="upload-compact-btn primary"
                      onClick={handleReset}
                    >
                      New Resume
                    </button>
                    <button
                      className="upload-compact-btn ghost"
                      onClick={handleTogglePreview}
                      disabled={!uploadedFile}
                    >
                      {showPreview ? 'Close PDF' : 'View PDF'}
                    </button>
                  </div>
                </div>
              ) : (
                /* Hero mode: full centered upload */
                <>
                  <div className="upload-hero-header">
                    <div className="upload-icon-wrapper">
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="#2563eb">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm4 18H6V4h7v5h5v11z" />
                      </svg>
                    </div>
                    <div className="upload-hero-text">
                      <h3>Upload Your Resume</h3>
                      <p>Drop your PDF resume below to start the analysis</p>
                    </div>
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
                      {hasFile ? (
                        <div className="file-uploaded">
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

                  <div className="upload-hero-actions">
                    <button
                      className="upload-button"
                      onClick={() => {
                        if (!uploadedFile) {
                          fileInputRef.current?.click();
                        } else {
                          handleAnalyze();
                        }
                      }}
                      disabled={isLoading}
                    >
                      {isLoading ? 'Analyzing...' : uploadedFile ? 'Upload and Analyze' : 'Select Resume'}
                    </button>
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
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="#2563eb">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
                      </svg>
                    </span>
                    <p>Please remove sensitive information (e.g., address, SSN, personal contact details) before uploading your resume.</p>
                  </div>
                </>
              )}
            </div>
          </section>

          {/* ── Results Area ───────────────────────────────────── */}
          {isLoading && (
            <section className="results-area">
              <div className="loading-section">
                <div className="loading-spinner"></div>
                <h3>Analyzing Your Resume...</h3>
                <p>Please wait while our AI processes your information</p>
              </div>
            </section>
          )}

          {showPreview && !isLoading && (
            <section className="results-area">
              <div className={`preview-panel ${isClosing ? 'closing' : ''}`}>
                <div className="preview-panel-header">
                  <h3>Resume Preview</h3>
                  <button className="close-preview-btn" onClick={handleClosePreview} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    {React.createElement(otherIcons["FaTimes"])} Close
                  </button>
                </div>
                <div className="preview-content-full">
                  <iframe
                    src={previewUrl}
                    className="preview-iframe-full"
                    title="Resume Preview"
                  />
                </div>
              </div>
            </section>
          )}

          {showResults && !isLoading && !showPreview && predictionResults && predictionResults.top_predictions && predictionResults.top_predictions.length > 0 && (
            <section className="results-area">
              {/* Results Header */}
              <div className="results-header-bar">
                <div>
                  <h3>Your Career Analysis</h3>
                  <p className="results-context">
                    These results compare your profile against 26 career paths. The "Profile Fit" score reflects alignment strength, not probability of success.
                  </p>
                </div>
              </div>

              {/* Low-confidence advisory note */}
              {(() => {
                const topRaw = predictionResults.top_predictions[0]?.raw_confidence ?? 0;
                if (topRaw < 5) {
                  return (
                    <div className="confidence-advisory confidence-advisory--warning">
                      <span className="confidence-advisory-icon">
                        {React.createElement(otherIcons["FaExclamationTriangle"], { size: 16 })}
                      </span>
                      <div className="confidence-advisory-body">
                        <strong>Low-signal prediction</strong>
                        <p>The model found very few distinguishing patterns in your resume (below 35% Profile Fit). At this level the prediction is essentially a guess. Try uploading a more detailed or role-specific resume for better results.</p>
                      </div>
                    </div>
                  );
                }
                if (topRaw < 10) {
                  return (
                    <div className="confidence-advisory confidence-advisory--caution">
                      <span className="confidence-advisory-icon">
                        {React.createElement(otherIcons["FaExclamationTriangle"], { size: 16 })}
                      </span>
                      <div className="confidence-advisory-body">
                        <strong>Uncertain prediction</strong>
                        <p>The model's confidence is low (below 49% Profile Fit). At this level, predictions are correct roughly half the time. Consider adding more work experience or skills detail to your resume.</p>
                      </div>
                    </div>
                  );
                }
                return null;
              })()}

              {/* Primary Match — Full Width Hero Card */}
              {(() => {
                const top = predictionResults.top_predictions[0];
                const reliability = calculateProfileFit(top.raw_confidence);
                return (
                  <div className="primary-prediction-card">
                    <div className="primary-card-header">
                      <div className="primary-icon-wrapper">
                        {getCareerIcon(top.career_path)}
                      </div>
                      <div className="primary-info">
                        <h4 className="section-label">Strongest Match</h4>
                        <h3 className="primary-career-name">{top.career_path}</h3>
                        <span className="primary-label">Rank #1 based on your profile</span>
                      </div>
                      <div className="primary-fit-badge">
                        <span className="fit-number">{reliability}%</span>
                        <span className="fit-label">Profile Fit</span>
                        <span className="fit-raw">({top.raw_confidence.toFixed(1)}% raw)</span>
                      </div>
                    </div>

                    <div className="primary-profile-fit">
                      <div className="profile-fit-bar">
                        <div className="profile-fit-bar-fill" style={{ width: `${reliability}%` }}></div>
                      </div>
                    </div>

                    <button
                      className="learn-more-button"
                      onClick={() => handleViewDetailedAnalysis(top.career_path)}
                    >
                      <span>View Detailed Analysis</span>
                      {React.createElement(otherIcons["FaArrowRight"], { size: 14 })}
                    </button>
                  </div>
                );
              })()}

              {/* Secondary Matches — Side by Side Cards */}
              {predictionResults.top_predictions.length > 1 && (
                <div className="secondary-matches-grid">
                  <h4 className="section-label">Also Strong Matches</h4>
                  <div className="secondary-cards-row">
                    {predictionResults.top_predictions.slice(1, 3).map((career, index) => {
                      const fit = calculateProfileFit(career.raw_confidence);
                      return (
                        <div key={index + 1} className="secondary-prediction-card">
                          <div className="secondary-card-top">
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
                                {fit}%
                                <span style={{ fontSize: '0.7rem', color: '#9ca3af', marginLeft: '4px', fontWeight: 'normal' }}>
                                  ({career.raw_confidence.toFixed(1)}% raw)
                                </span>
                              </span>
                            </div>
                            <div className="secondary-fit-bar">
                              <div className="secondary-fit-bar-fill" style={{ width: `${fit}%` }}></div>
                            </div>
                          </div>

                          <button
                            className="secondary-view-button"
                            onClick={() => handleViewDetailedAnalysis(career.career_path)}
                          >
                            View Analysis
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Full Transparency Section */}
              <div className="transparency-section">
                <button
                  className="expand-button"
                  onClick={() => setShowAllPaths(!showAllPaths)}
                >
                  {showAllPaths ? "Hide All Career Paths" : "View All Career Paths"}
                </button>

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
            </section>
          )}

        </div>
      </main>

      {/* Career Paths Modal */}
      {showCareerPaths && (
        <CareerPathsModal onClose={() => setShowCareerPaths(false)} />
      )}

      {/* Change Password Modal */}
      <ChangePasswordModal
        isOpen={showChangePassword}
        onClose={() => setShowChangePassword(false)}
      />
    </div>
  );
};

export default Dashboard;
