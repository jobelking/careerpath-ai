import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../../components/common/Logo';
import './History.css';

const History = () => {
  const navigate = useNavigate();
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);

  // Mock history data
  const analysisHistory = [
    {
      id: 1,
      date: '2024-12-20',
      time: '10:30 AM',
      fileName: 'resume_software_engineer_2024.pdf',
      topCareer: 'Software Engineer',
      matchScore: 92,
      status: 'completed',
      predictions: [
        { name: 'Software Engineer', percentage: 92, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M20 18c1.1 0 1.99-.9 1.99-2L22 6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2H0v2h24v-2h-4zM4 6h16v10H4V6z"/></svg> },
        { name: 'Data Scientist', percentage: 85, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg> },
        { name: 'Business Analyst', percentage: 78, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M20 6h-4V4c0-1.11-.89-2-2-2h-4c-1.11 0-2 .89-2 2v2H4c-1.11 0-1.99.89-1.99 2L2 19c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2zm-6 0h-4V4h4v2z"/></svg> },
        { name: 'Product Manager', percentage: 72, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M17 1.01L7 1c-1.1 0-2 .9-2 2v18c0 1.1.9 2 2 2h10c1.1 0 2-.9 2-2V3c0-1.1-.9-1.99-2-1.99zM17 19H7V5h10v14z"/></svg> },
        { name: 'UI/UX Designer', percentage: 68, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/></svg> }
      ]
    },
    {
      id: 2,
      date: '2024-12-15',
      time: '02:15 PM',
      fileName: 'updated_resume_v2.pdf',
      topCareer: 'Data Scientist',
      matchScore: 88,
      status: 'completed',
      predictions: [
        { name: 'Data Scientist', percentage: 88, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg> },
        { name: 'Machine Learning Engineer', percentage: 84, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M20 9V7c0-1.1-.9-2-2-2h-3c0-1.66-1.34-3-3-3S9 3.34 9 5H6c-1.1 0-2 .9-2 2v2c-1.66 0-3 1.34-3 3s1.34 3 3 3v4c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-4c1.66 0 3-1.34 3-3s-1.34-3-3-3zm-2 10H6V7h12v12z"/></svg> },
        { name: 'Software Engineer', percentage: 81, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M20 18c1.1 0 1.99-.9 1.99-2L22 6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2H0v2h24v-2h-4zM4 6h16v10H4V6z"/></svg> },
        { name: 'Research Analyst', percentage: 75, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M20.5 3l-.16.03L15 5.1 9 3 3.36 4.9c-.21.07-.36.25-.36.48V20.5c0 .28.22.5.5.5l.16-.03L9 18.9l6 2.1 5.64-1.9c.21-.07.36-.25.36-.48V3.5c0-.28-.22-.5-.5-.5zM15 19l-6-2.11V5l6 2.11V19z"/></svg> },
        { name: 'Business Intelligence', percentage: 70, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M3.5 18.49l6-6.01 4 4L22 6.92l-1.41-1.41-7.09 7.97-4-4L2 16.99z"/></svg> }
      ]
    },
    {
      id: 3,
      date: '2024-12-10',
      time: '09:45 AM',
      fileName: 'john_doe_resume.pdf',
      topCareer: 'Product Manager',
      matchScore: 85,
      status: 'completed',
      predictions: [
        { name: 'Product Manager', percentage: 85, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M17 1.01L7 1c-1.1 0-2 .9-2 2v18c0 1.1.9 2 2 2h10c1.1 0 2-.9 2-2V3c0-1.1-.9-1.99-2-1.99zM17 19H7V5h10v14z"/></svg> },
        { name: 'Project Manager', percentage: 82, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm2 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg> },
        { name: 'Business Analyst', percentage: 79, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M20 6h-4V4c0-1.11-.89-2-2-2h-4c-1.11 0-2 .89-2 2v2H4c-1.11 0-1.99.89-1.99 2L2 19c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2zm-6 0h-4V4h4v2z"/></svg> },
        { name: 'Consultant', percentage: 74, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7z"/></svg> },
        { name: 'Marketing Manager', percentage: 69, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg> }
      ]
    },
    {
      id: 4,
      date: '2024-12-05',
      time: '04:20 PM',
      fileName: 'resume_final_draft.pdf',
      topCareer: 'UI/UX Designer',
      matchScore: 90,
      status: 'completed',
      predictions: [
        { name: 'UI/UX Designer', percentage: 90, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/></svg> },
        { name: 'Product Designer', percentage: 87, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M12 2l-5.5 9h11z"/><circle cx="17.5" cy="17.5" r="4.5"/><path d="M3 13.5h8v8H3z"/></svg> },
        { name: 'Graphic Designer', percentage: 82, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M17.5 8c1.9 0 3.5-1.6 3.5-3.5S19.4 1 17.5 1 14 2.6 14 4.5 15.6 8 17.5 8zm-11 0C8.4 8 10 6.4 10 4.5S8.4 1 6.5 1 3 2.6 3 4.5 4.6 8 6.5 8zM6.5 10C4 10 1 11.3 1 13.2V16h11v-2.8c0-1.9-3-3.2-5.5-3.2zm11 0c-2.5 0-5.5 1.3-5.5 3.2V16h11v-2.8c0-1.9-3-3.2-5.5-3.2zM12 18H1v3h11v-3zm11 0H12v3h11v-3z"/></svg> },
        { name: 'Front-End Developer', percentage: 78, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M20 18c1.1 0 1.99-.9 1.99-2L22 6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2H0v2h24v-2h-4zM4 6h16v10H4V6z"/></svg> },
        { name: 'Creative Director', percentage: 73, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg> }
      ]
    },
    {
      id: 5,
      date: '2024-11-28',
      time: '11:00 AM',
      fileName: 'professional_resume.pdf',
      topCareer: 'Business Analyst',
      matchScore: 86,
      status: 'completed',
      predictions: [
        { name: 'Business Analyst', percentage: 86, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M20 6h-4V4c0-1.11-.89-2-2-2h-4c-1.11 0-2 .89-2 2v2H4c-1.11 0-1.99.89-1.99 2L2 19c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2zm-6 0h-4V4h4v2z"/></svg> },
        { name: 'Data Analyst', percentage: 83, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg> },
        { name: 'Financial Analyst', percentage: 79, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M11.8 10.9c-2.27-.59-3-1.2-3-2.15 0-1.09 1.01-1.85 2.7-1.85 1.78 0 2.44.85 2.5 2.1h2.21c-.07-1.72-1.12-3.3-3.21-3.81V3h-3v2.16c-1.94.42-3.5 1.68-3.5 3.61 0 2.31 1.91 3.46 4.7 4.13 2.5.6 3 1.48 3 2.41 0 .69-.49 1.79-2.7 1.79-2.06 0-2.87-.92-2.98-2.1h-2.2c.12 2.19 1.76 3.42 3.68 3.83V21h3v-2.15c1.95-.37 3.5-1.5 3.5-3.55 0-2.84-2.43-3.81-4.7-4.4z"/></svg> },
        { name: 'Operations Manager', percentage: 75, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94L14.4 2.81c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/></svg> },
        { name: 'Strategy Consultant', percentage: 71, icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg> }
      ]
    }
  ];

  const handleViewDetails = (analysis) => {
    setSelectedAnalysis(selectedAnalysis?.id === analysis.id ? null : analysis);
  };

  const handleNewAnalysis = () => {
    navigate('/dashboard');
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return date.toLocaleDateString('en-US', options);
  };

  return (
    <div className="history-container">
      {/* Header/Navigation */}
      <header className="history-header">
        <div className="header-content">
          <h1 className="history-brand" onClick={() => navigate('/')}>
            <Logo variant='modern' />
          </h1>
          <nav className="history-nav">
            <button className="nav-link" onClick={() => navigate('/dashboard')}>Dashboard</button>
            <button className="nav-link active">History</button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="history-main">
        <div className="history-content">
          {/* Page Header */}
          <div className="page-header">
            <div className="page-header-text">
              <h2>Analysis History</h2>
              <p>View and compare your past career predictions</p>
            </div>
            <button className="new-analysis-button" onClick={handleNewAnalysis}>
              <span className="button-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
                </svg>
              </span>
              New Analysis
            </button>
          </div>

          {/* History Grid */}
          <div className="history-grid">
            {analysisHistory.length > 0 ? (
              <div className="history-list">
                {analysisHistory.map((analysis) => (
                  <div key={analysis.id} className="history-item">
                    <div className="history-card" onClick={() => handleViewDetails(analysis)}>
                      <div className="card-header">
                        <div className="card-header-left">
                          <div className="file-icon-badge">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="#2563eb">
                              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm4 18H6V4h7v5h5v11z"/>
                            </svg>
                          </div>
                          <div className="file-info">
                            <h3 className="file-name">{analysis.fileName}</h3>
                            <p className="file-date">
                              {formatDate(analysis.date)} at {analysis.time}
                            </p>
                          </div>
                        </div>
                        <div className="card-header-right">
                          <div className="status-badge">{analysis.status}</div>
                          <button className="expand-button">
                            {selectedAnalysis?.id === analysis.id ? '▲' : '▼'}
                          </button>
                        </div>
                      </div>

                      <div className="card-summary">
                        <div className="summary-item">
                          <span className="summary-label">Top Match</span>
                          <span className="summary-value">{analysis.topCareer}</span>
                        </div>
                        <div className="summary-divider"></div>
                        <div className="summary-item">
                          <span className="summary-label">Match Score</span>
                          <span className="summary-value highlight">{analysis.matchScore}%</span>
                        </div>
                      </div>
                    </div>

                    {/* Expanded Details */}
                    <div className={`expanded-details ${selectedAnalysis?.id === analysis.id ? 'open' : ''}`}>
                        <div className="details-header">
                          <h4>Career Predictions</h4>
                          <p>All predictions from this analysis</p>
                        </div>
                        <div className="details-predictions">
                          {analysis.predictions.map((career, index) => (
                            <div key={index} className="detail-prediction-card">
                              <div className="detail-prediction-left">
                                <div className="detail-prediction-rank">#{index + 1}</div>
                                <div className="detail-prediction-icon">{career.icon}</div>
                                <div className="detail-prediction-info">
                                  <h5 className="detail-prediction-name">{career.name}</h5>
                                  <span className="detail-prediction-match">Match Score</span>
                                </div>
                              </div>
                              <div className="detail-prediction-right">
                                <div className="detail-prediction-percentage">{career.percentage}%</div>
                                <div className="detail-prediction-bar-container">
                                  <div 
                                    className="detail-prediction-bar-fill" 
                                    style={{ width: `${career.percentage}%` }}
                                  ></div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-history">
                <div className="empty-icon">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="#2563eb">
                    <path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/>
                  </svg>
                </div>
                <h3>No Analysis History</h3>
                <p>You haven't analyzed any resumes yet. Start your first analysis to see results here.</p>
                <button className="start-button" onClick={handleNewAnalysis}>
                  Start New Analysis
                </button>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default History;
