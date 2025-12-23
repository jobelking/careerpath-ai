import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';

const Dashboard = () => {
  const navigate = useNavigate();
  const [uploadedFile, setUploadedFile] = useState(null);
  const [showResults, setShowResults] = useState(false);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
      // Simulate analysis
      setTimeout(() => {
        setShowResults(true);
      }, 1500);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setUploadedFile(file);
      setTimeout(() => {
        setShowResults(true);
      }, 1500);
    }
  };

  // Mock career predictions data
  const careerPredictions = [
    { name: 'Software Engineer', percentage: 92, icon: '💻' },
    { name: 'Data Scientist', percentage: 85, icon: '📊' },
    { name: 'Business Analyst', percentage: 78, icon: '💼' },
    { name: 'Product Manager', percentage: 72, icon: '📱' },
    { name: 'UI/UX Designer', percentage: 68, icon: '🎨' }
  ];

  return (
    <div className="dashboard-container">
      {/* Header/Navigation */}
      <header className="dashboard-header">
        <div className="header-content">
          <h1 className="dashboard-brand" onClick={() => navigate('/')}>CareerPath-AI</h1>
          <nav className="dashboard-nav">
            <button className="nav-link active">Dashboard</button>
            <button className="nav-link">History</button>
            <button className="nav-link">Settings</button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="dashboard-main">
        <div className="dashboard-content">
          {/* Welcome Section */}
          <div className="welcome-section">
            <h2>Welcome to Your Career Dashboard</h2>
            <p>Upload your resume to discover your ideal career path with AI-powered analysis</p>
          </div>

          {/* Two-Column Layout */}
          <div className="dashboard-grid">
            {/* Left Column - Upload Section */}
            <div className="left-column">
              <div className="upload-card">
                <div className="upload-header">
                  <div className="upload-icon-wrapper">
                    <span className="upload-icon">📄</span>
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
                        <span className="file-icon">✓</span>
                        <span className="file-name">{uploadedFile.name}</span>
                        <span className="analyzing-text">Analyzing...</span>
                      </div>
                    ) : (
                      <div className="upload-prompt">
                        <div className="upload-cloud">☁️</div>
                        <span className="upload-text">Click to upload or drag and drop</span>
                        <span className="upload-formats">PDF, DOC, DOCX (Max 10MB)</span>
                      </div>
                    )}
                  </label>
                </div>
                
                {/* Privacy Note */}
                <div className="privacy-note">
                  <span className="info-icon">ℹ️</span>
                  <p>Please remove sensitive information (e.g., address, SSN, personal contact details) before uploading your resume.</p>
                </div>
              </div>

              {/* Analysis Info Card */}
              <div className="info-card">
                <h4>How It Works</h4>
                <ul className="info-list">
                  <li>
                    <span className="step-number">1</span>
                    <span>Upload your resume in PDF, DOC, or DOCX format</span>
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
              {showResults ? (
                <div className="results-section">
                  <div className="results-header">
                    <h3>Your Top Career Matches</h3>
                    <p>Based on your skills, experience, and qualifications</p>
                  </div>

                  <div className="predictions-list">
                    {careerPredictions.map((career, index) => (
                      <div key={index} className="prediction-card" style={{ animationDelay: `${index * 0.1}s` }}>
                        <div className="prediction-left">
                          <div className="prediction-rank">#{index + 1}</div>
                          <div className="prediction-icon">{career.icon}</div>
                          <div className="prediction-info">
                            <h4 className="prediction-name">{career.name}</h4>
                            <span className="prediction-match">Match Score</span>
                          </div>
                        </div>
                        <div className="prediction-right">
                          <div className="prediction-percentage">{career.percentage}%</div>
                          <div className="prediction-bar-container">
                            <div 
                              className="prediction-bar-fill" 
                              style={{ width: `${career.percentage}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="placeholder-section">
                  <div className="placeholder-icon">📊</div>
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
