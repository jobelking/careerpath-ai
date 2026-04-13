import React from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../../components/common/Logo';
import { useAuth } from '../../context/AuthContext';
import AuthModal from '../../components/auth/AuthModal';
import { otherIcons } from '../../utils/otherIcons';
import logoutIcon from '../../styles/logout.svg';
import './LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();
  const { currentUser, logout } = useAuth();
  const [isAuthModalOpen, setIsAuthModalOpen] = React.useState(false);
  const [authModalView, setAuthModalView] = React.useState('login');
  const [showLogoutConfirm, setShowLogoutConfirm] = React.useState(false);

  const handleGetStarted = () => {
    if (currentUser) {
      navigate('/dashboard');
    } else {
      setAuthModalView('register');
      setIsAuthModalOpen(true);
    }
  };

  const handleLoginClick = () => {
    setAuthModalView('login');
    setIsAuthModalOpen(true);
  };

  const handleRegisterClick = () => {
    setAuthModalView('register');
    setIsAuthModalOpen(true);
  };

  const handleLogout = () => {
    logout();
    setShowLogoutConfirm(false);
  };

  const userDisplayName =
    currentUser?.fullName ||
    currentUser?.name ||
    (currentUser?.email ? currentUser.email.split('@')[0] : 'User');

  return (
    <div className="landing-container">

      {/* Constellation Background */}
      <div className="constellation" aria-hidden="true">
        <div className="constellation-orb" />
        <div className="constellation-orb" />
        <div className="constellation-orb" />
        <div className="constellation-orb" />
      </div>

      {/* Grain Overlay */}
      <div className="landing-grain" aria-hidden="true" />


      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="landing-header">
        <div className="header-logo">
          <Logo variant="modern" />
        </div>
        <div className="header-actions">
          {!currentUser ? (
            <div className="auth-buttons">
              <button className="auth-btn login-btn" onClick={handleLoginClick}>Login</button>
              <button className="auth-btn register-btn" onClick={handleRegisterClick}>Get Started</button>
            </div>
          ) : (
            <div className="user-profile-menu">
              <span className="user-greeting">Hi, {userDisplayName}</span>
              <button className="auth-btn dashboard-btn" onClick={() => navigate('/dashboard')}>Dashboard</button>
              <button className="auth-btn logout" onClick={() => setShowLogoutConfirm(true)} aria-label="Logout" title="Logout">
                <img src={logoutIcon} alt="" className="logout-icon" />
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Logout Confirmation */}
      {showLogoutConfirm && (
        <div className="landing-logout-bar">
          <span>Are you sure you want to logout?</span>
          <div className="landing-logout-actions">
            <button className="landing-logout-confirm" onClick={handleLogout}>Yes, Logout</button>
            <button className="landing-logout-cancel" onClick={() => setShowLogoutConfirm(false)}>Cancel</button>
          </div>
        </div>
      )}

      {/* ── Hero Section ────────────────────────────────────────── */}
      <main className="hero-section">
        <div className="hero-content">
          <div className="hero-text">


            <h1 className="brand-name">
              Find your best&#8209;fit{' '}
              <span className="accent">career path</span>, faster.
            </h1>

            <h2 className="hero-tagline">
              Upload your resume and discover personalized career recommendations in seconds.
            </h2>

            <p className="hero-description">
              Our <strong>machine learning</strong> engine analyzes your skills, experience, and qualifications,
              then matches you to the most suitable paths across <strong>26 career categories</strong>.
            </p>

            <div className="hero-cta-group">
              {!currentUser ? (
                <>
                  <button className="cta-primary" onClick={handleGetStarted}>
                    Start Free Analysis
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                  </button>
                  <button className="cta-ghost" onClick={handleLoginClick}>
                    I have an account
                  </button>
                </>
              ) : (
                <button className="cta-primary" onClick={() => navigate('/dashboard')}>
                  Go to Dashboard
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </button>
              )}
            </div>
          </div>

          {/* Hero Visual — Career Results Preview */}
          <div className="hero-visual">
            <div className="career-results-card">
              <div className="card-header">
                <div className="status-indicator">
                  <span className="status-dot red" />
                  <span className="status-dot yellow" />
                  <span className="status-dot green" />
                  <span className="status-text">Analysis Complete</span>
                </div>
                <div className="file-info">
                  <span className="file-name">analysis_v2.0.pdf</span>
                </div>
              </div>

              <div className="card-title">
                <h2>Top 3 Career Path Matches</h2>
                <span className="analysis-id">ID: #8X29</span>
              </div>

              <div className="matches-container">
                {/* Primary Match */}
                <div className="match-card primary">
                  <div className="match-content">
                    <div className="match-percentage">
                      <svg viewBox="0 0 120 120" className="circular-progress">
                        <circle className="progress-bg" cx="60" cy="60" r="50" />
                        <circle className="progress-bar" cx="60" cy="60" r="50"
                          style={{ strokeDashoffset: 'calc(314 - (314 * 87) / 100)' }} />
                        <text x="60" y="60" className="percentage-text">87%</text>
                      </svg>
                    </div>
                    <div className="match-details">
                      <h3 className="match-title">Banking &amp; Financial Services</h3>
                      <div className="match-tags">
                        <span className="tag high-demand">High Demand</span>
                        <span className="tag">Finance + Ops</span>
                      </div>
                    </div>
                  </div>
                  <button className="match-arrow">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>

                {/* Secondary Match 1 */}
                <div className="match-card secondary">
                  <div className="match-content">
                    <div className="match-percentage small">
                      <svg viewBox="0 0 100 100" className="circular-progress">
                        <circle className="progress-bg" cx="50" cy="50" r="40" />
                        <circle className="progress-bar" cx="50" cy="50" r="40"
                          style={{ strokeDashoffset: 'calc(251 - (251 * 74) / 100)' }} />
                        <text x="50" y="50" className="percentage-text">74%</text>
                      </svg>
                    </div>
                    <div className="match-details">
                      <h3 className="match-title">Finance &amp; Investment</h3>
                      <div className="match-tags">
                        <span className="tag">Analysis + Strategy</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Secondary Match 2 */}
                <div className="match-card secondary">
                  <div className="match-content">
                    <div className="match-percentage small">
                      <svg viewBox="0 0 100 100" className="circular-progress">
                        <circle className="progress-bg" cx="50" cy="50" r="40" />
                        <circle className="progress-bar" cx="50" cy="50" r="40"
                          style={{ strokeDashoffset: 'calc(251 - (251 * 62) / 100)' }} />
                        <text x="50" y="50" className="percentage-text">62%</text>
                      </svg>
                    </div>
                    <div className="match-details">
                      <h3 className="match-title">Cybersecurity</h3>
                      <div className="match-tags">
                        <span className="tag">Security + Tech</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="processing-badge">
                <div className="badge-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M7 2v11h3v9l7-12h-4l4-8z" />
                  </svg>
                </div>
                <div className="badge-text">
                  <span className="badge-label">Processing</span>
                  <span className="badge-value">0.42 seconds</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ── How It Works ────────────────────────────────────────── */}
      <section className="steps-section">
        <div className="steps-header">
          <span className="steps-label">How It Works</span>
          <h2>Three steps to your ideal career</h2>
        </div>

        <div className="steps-grid">
          <div className="step-card">
            <div className="step-connector" />
            <div className="step-number">1</div>
            <h3>Upload Your Resume</h3>
            <p>Drop in your PDF or text resume — we accept all standard formats securely.</p>
          </div>

          <div className="step-card">
            <div className="step-connector" />
            <div className="step-number">2</div>
            <h3>AI Analyzes Your Profile</h3>
            <p>Our trained model extracts skills and experience to match against career patterns.</p>
          </div>

          <div className="step-card">
            <div className="step-number">3</div>
            <h3>Get Career Matches</h3>
            <p>Receive ranked career paths with confidence scores and actionable learning roadmaps.</p>
          </div>
        </div>
      </section>

      {/* ── Features Grid ───────────────────────────────────────── */}
      <section className="features-section">
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">
              {React.createElement(otherIcons["FaBolt"], { size: 22 })}
            </div>
            <h3 className="feature-title">Lightning Fast</h3>
            <p className="feature-description">Career predictions in seconds with our optimized AI engine</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">
              {React.createElement(otherIcons["FaCheckCircle"], { size: 22 })}
            </div>
            <h3 className="feature-title">Highly Accurate</h3>
            <p className="feature-description">Advanced machine learning ensures precise career matching</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">
              {React.createElement(otherIcons["FaLock"], { size: 22 })}
            </div>
            <h3 className="feature-title">Secure &amp; Private</h3>
            <p className="feature-description">Resume data encrypted and never shared with third parties</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">
              {React.createElement(otherIcons["MdInsights"], { size: 26 })}
            </div>
            <h3 className="feature-title">Deep Insights</h3>
            <p className="feature-description">Skill matching, career probabilities, and learning roadmaps</p>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────── */}
      <footer className="landing-footer">
        <p>&copy; 2025 CareerPath-AI. Empowering career decisions with artificial intelligence.</p>
      </footer>

      {/* Auth Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        initialView={authModalView}
      />
    </div>
  );
};

export default LandingPage;
