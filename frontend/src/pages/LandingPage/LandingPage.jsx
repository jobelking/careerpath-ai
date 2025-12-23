import React from 'react';
import { useNavigate } from 'react-router-dom';
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
            <h1 className="hero-title">
              <span className="brand-name">CareerPath-AI</span>
            </h1>
            <p className="hero-tagline">
              Discover Your Perfect Career Path with AI-Powered Resume Analysis
            </p>
            <p className="hero-description" style={{ textAlign: 'justify' }}>
              Upload your resume and let our advanced AI technology analyze your skills, 
              experience, and qualifications to recommend the most suitable career paths tailored just for you.
            </p>
            <button className="cta-button" onClick={handleGetStarted}>
              <span className="cta-text">Upload Resume & Get Started</span>
              <span className="cta-icon">→</span>
            </button>
          </div>

          <div className="hero-visual">
            <div className="ai-illustration">
              <div className="visual-container">
                <div className="resume-preview">
                  <div className="resume-header">
                    <div className="resume-photo"></div>
                    <div className="resume-text-lines">
                      <div className="text-line long"></div>
                      <div className="text-line medium"></div>
                      <div className="text-line short"></div>
                    </div>
                  </div>
                  <div className="resume-body">
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line"></div>
                    <div className="text-line medium"></div>
                  </div>
                </div>
                
                <div className="ai-processor">
                  <div className="processor-circle">
                    <div className="orbit orbit-1">
                      <div className="orbit-dot"></div>
                    </div>
                    <div className="orbit orbit-2">
                      <div className="orbit-dot"></div>
                    </div>
                    <div className="orbit orbit-3">
                      <div className="orbit-dot"></div>
                    </div>
                    <div className="ai-core">AI</div>
                  </div>
                </div>
                
                <div className="career-output">
                  <div className="output-card">
                    <div className="output-icon">💼</div>
                    <span>Business</span>
                  </div>
                  <div className="output-card">
                    <div className="output-icon">💻</div>
                    <span>Tech</span>
                  </div>
                  <div className="output-card">
                    <div className="output-icon">🎨</div>
                    <span>Creative</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3 className="feature-title">Lightning Fast</h3>
            <p className="feature-description">Get career predictions in seconds with our optimized AI engine</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🎯</div>
            <h3 className="feature-title">Highly Accurate</h3>
            <p className="feature-description">Advanced machine learning ensures precise career recommendations</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🔒</div>
            <h3 className="feature-title">Secure & Private</h3>
            <p className="feature-description">Your resume data is encrypted and never shared with third parties</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3 className="feature-title">Detailed Insights</h3>
            <p className="feature-description">Comprehensive analysis with skill matching and career probabilities</p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="landing-footer">
        <p>© 2025 CareerPath-AI. Empowering your career decisions with artificial intelligence.</p>
      </footer>
    </div>
  );
};

export default LandingPage;
