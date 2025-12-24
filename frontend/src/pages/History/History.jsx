import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
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
        { name: 'Software Engineer', percentage: 92, icon: '💻' },
        { name: 'Data Scientist', percentage: 85, icon: '📊' },
        { name: 'Business Analyst', percentage: 78, icon: '💼' },
        { name: 'Product Manager', percentage: 72, icon: '📱' },
        { name: 'UI/UX Designer', percentage: 68, icon: '🎨' }
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
        { name: 'Data Scientist', percentage: 88, icon: '📊' },
        { name: 'Machine Learning Engineer', percentage: 84, icon: '🤖' },
        { name: 'Software Engineer', percentage: 81, icon: '💻' },
        { name: 'Research Analyst', percentage: 75, icon: '🔬' },
        { name: 'Business Intelligence', percentage: 70, icon: '📈' }
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
        { name: 'Product Manager', percentage: 85, icon: '📱' },
        { name: 'Project Manager', percentage: 82, icon: '📋' },
        { name: 'Business Analyst', percentage: 79, icon: '💼' },
        { name: 'Consultant', percentage: 74, icon: '💡' },
        { name: 'Marketing Manager', percentage: 69, icon: '📢' }
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
        { name: 'UI/UX Designer', percentage: 90, icon: '🎨' },
        { name: 'Product Designer', percentage: 87, icon: '✨' },
        { name: 'Graphic Designer', percentage: 82, icon: '🖌️' },
        { name: 'Front-End Developer', percentage: 78, icon: '💻' },
        { name: 'Creative Director', percentage: 73, icon: '🎭' }
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
        { name: 'Business Analyst', percentage: 86, icon: '💼' },
        { name: 'Data Analyst', percentage: 83, icon: '📊' },
        { name: 'Financial Analyst', percentage: 79, icon: '💰' },
        { name: 'Operations Manager', percentage: 75, icon: '⚙️' },
        { name: 'Strategy Consultant', percentage: 71, icon: '🎯' }
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
          <h1 className="history-brand" onClick={() => navigate('/')}>CareerPath-AI</h1>
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
              <span className="button-icon">➕</span>
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
                          <div className="file-icon-badge">📄</div>
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
                <div className="empty-icon">📭</div>
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
