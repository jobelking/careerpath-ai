import React from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../../components/common/Logo';
import { careerIcons } from '../../utils/careerIcons';
import { otherIcons } from '../../utils/otherIcons';
import './LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/dashboard');
  };

  return (
    <div className="landing-container">
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-text">
            <Logo variant="modern" className="brand-name" />
            <p className="hero-tagline">
              Discover Your Perfect Career Path with AI-Powered Resume Analysis
            </p>
            <p className="hero-description" style={{ textAlign: 'justify' }}>
              Upload your resume and let our advanced AI technology analyze your skills, 
              experience, and qualifications to recommend the most suitable career paths tailored just for you.
            </p>
            <button className="cta-button" onClick={handleGetStarted}>
              <span className="cta-text">Upload Resume & Get Started</span>
              <span className="cta-icon">➜</span>
            </button>
          </div>

          <div className="hero-visual">
            <div className="career-results-card">
              <div className="card-header">
                <div className="status-indicator">
                  <span className="status-dot red"></span>
                  <span className="status-dot yellow"></span>
                  <span className="status-dot green"></span>
                  <span className="status-text">ANALYSIS COMPLETE</span>
                </div>
                <div className="file-info">
                  <span className="file-name">analysis_v2.0.pdf</span>
                </div>
              </div>
              
              <div className="card-title">
                <h2>Top 3 Career Matches</h2>
                <span className="analysis-id">ID: #8X29</span>
              </div>
              
              <div className="matches-container">
                <div className="match-card primary">
                  <div className="match-content">
                    <div className="match-percentage">
                      <svg viewBox="0 0 120 120" className="circular-progress">
                        <circle className="progress-bg" cx="60" cy="60" r="50" />
                        <circle className="progress-bar" cx="60" cy="60" r="50" 
                          style={{strokeDashoffset: 'calc(314 - (314 * 87) / 100)'}} />
                        <text x="60" y="60" className="percentage-text">87%</text>
                      </svg>
                    </div>
                    <div className="match-details">
                      <h3 className="match-title">
                        Banking Officer
                      </h3>
                      <div className="match-tags">
                        <span className="tag high-demand">High Demand</span>
                        <span className="tag">Finance + Investment</span>
                      </div>
                    </div>
                  </div>
                  <button className="match-arrow">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
                
                <div className="match-card secondary">
                  <div className="match-content">
                    <div className="match-percentage small">
                      <svg viewBox="0 0 100 100" className="circular-progress">
                        <circle className="progress-bg" cx="50" cy="50" r="40" />
                        <circle className="progress-bar" cx="50" cy="50" r="40" 
                          style={{strokeDashoffset: 'calc(251 - (251 * 74) / 100)'}} />
                        <text x="50" y="50" className="percentage-text">74%</text>
                      </svg>
                    </div>
                    <div className="match-details">
                      <h3 className="match-title">
                        Finance Analyst
                      </h3>
                      <div className="match-tags">
                        <span className="tag">Management + Strategy</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="match-card secondary">
                  <div className="match-content">
                    <div className="match-percentage small">
                      <svg viewBox="0 0 100 100" className="circular-progress">
                        <circle className="progress-bg" cx="50" cy="50" r="40" />
                        <circle className="progress-bar" cx="50" cy="50" r="40" 
                          style={{strokeDashoffset: 'calc(251 - (251 * 62) / 100)'}} />
                        <text x="50" y="50" className="percentage-text">62%</text>
                      </svg>
                    </div>
                    <div className="match-details">
                      <h3 className="match-title">
                        Cyber Security Specialist
                      </h3>
                      <div className="match-tags">
                        <span className="tag">Security + Protection</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="processing-badge">
                <div className="badge-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M7 2v11h3v9l7-12h-4l4-8z"/>
                  </svg>
                </div>
                <div className="badge-text">
                  <span className="badge-label">Processing time</span>
                  <span className="badge-value">0.42 seconds</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">
              {React.createElement(otherIcons["FaBolt"], { size: 32, color: "#2563eb" })}
            </div>
            <h3 className="feature-title">Lightning Fast</h3>
            <p className="feature-description">Get career predictions in seconds with our optimized AI engine</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">
              {React.createElement(otherIcons["FaCheckCircle"], { size: 32, color: "#2563eb"})}
            </div>
            <h3 className="feature-title">Highly Accurate</h3>
            <p className="feature-description">Advanced machine learning ensures precise career recommendations</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">
               {React.createElement(otherIcons["FaLock"], { size: 32, color: "#2563eb"})}
            </div>
            <h3 className="feature-title">Secure & Private</h3>
            <p className="feature-description">Your resume data is encrypted and never shared with third parties</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">
              {React.createElement(otherIcons["MdInsights"], { size: 40, color: "#2563eb"})}
            </div>
            <h3 className="feature-title">Detailed Insights</h3>
            <p className="feature-description">Comprehensive analysis with skill matching and career probabilities</p>
          </div>
        </div>
      </div>

      {/* Footer */}
      {/* <footer className="landing-footer">
        <p>© 2025 CareerPath-AI. Empowering your career decisions with artificial intelligence.</p>
      </footer> */}
    </div>
  );
};

export default LandingPage;
