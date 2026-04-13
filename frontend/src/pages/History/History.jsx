import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { exportToPdf } from '../../utils/exportToPdf';
import Logo from '../../components/common/Logo';
import apiService from '../../services/api/apiService';
import { useAuth } from '../../context/AuthContext';
import { useDashboard } from '../../context/DashboardContext';
import { otherIcons } from '../../utils/otherIcons';
import CareerPathsModal from '../../components/common/CareerPathsModal/CareerPathsModal';
import './History.css';

const History = () => {
    const RECORDS_PER_PAGE = 10;
    const navigate = useNavigate();
    const { currentUser, logout, getToken } = useAuth();
    const {
        setPredictionResults,
        setResumeText,
        setLearningRoadmapByPath,
        setCertificationDataByPath,
        setSkillsInsightsByPath,
        setHistoryRecordId,
    } = useDashboard();
    const [history, setHistory] = useState([]);
    const [currentPage, setCurrentPage] = useState(1);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
    const [exportingId, setExportingId] = useState(null);
    const [exportMenuOpenId, setExportMenuOpenId] = useState(null);
    const [analysisMenuOpenId, setAnalysisMenuOpenId] = useState(null);
    const [menuOpen, setMenuOpen] = useState(false);
    const [showCareerPaths, setShowCareerPaths] = useState(false);
    const menuRef = useRef(null);

    // Close menu on outside click
    useEffect(() => {
        const handleOutsideClick = (e) => {
            if (menuRef.current && !menuRef.current.contains(e.target)) {
                setMenuOpen(false);
            }
        };
        if (menuOpen) document.addEventListener('mousedown', handleOutsideClick);
        return () => document.removeEventListener('mousedown', handleOutsideClick);
    }, [menuOpen]);

    useEffect(() => {
        const handleExportMenuClick = (e) => {
            if (!e.target.closest('.export-menu')) {
                setExportMenuOpenId(null);
            }
            if (!e.target.closest('.analysis-menu')) {
                setAnalysisMenuOpenId(null);
            }
        };
        document.addEventListener('mousedown', handleExportMenuClick);
        return () => document.removeEventListener('mousedown', handleExportMenuClick);
    }, []);

    // Close menu on route change / resize past breakpoint
    useEffect(() => {
        const handleResize = () => {
            if (window.innerWidth > 768) setMenuOpen(false);
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const handleLogout = () => {
        setMenuOpen(false);
        logout();
        navigate('/');
    };

    // ── Fetch history on mount ─────────────────────────────────────────────────
    useEffect(() => {
        const fetchHistory = async () => {
            setIsLoading(true);
            setError(null);
            try {
                const token = getToken();
                if (!token) {
                    navigate('/');
                    return;
                }
                const data = await apiService.getHistory(token);
                setHistory(data.history || []);
            } catch (err) {
                console.error('Failed to load history:', err);
                setError('Failed to load your prediction history. Please try again.');
            } finally {
                setIsLoading(false);
            }
        };
        fetchHistory();
    }, [getToken, navigate]);

    useEffect(() => {
        setCurrentPage(1);
    }, [history.length]);

    const totalPages = Math.max(1, Math.ceil(history.length / RECORDS_PER_PAGE));
    const startIndex = (currentPage - 1) * RECORDS_PER_PAGE;
    const paginatedHistory = history.slice(startIndex, startIndex + RECORDS_PER_PAGE);

    // ── Profile Fit (mirrors Dashboard logic exactly) ──────────────────────
    const calculateProfileFit = (rawProbability) => {
        const p = rawProbability;
        if (p < 5) return Math.round((p / 5) * 35);
        if (p < 10) return Math.round(35 + ((p - 5) / 5) * 14);
        if (p < 15) return Math.round(49 + ((p - 10) / 5) * 13);
        if (p < 20) return Math.round(62 + ((p - 15) / 5) * 18);
        if (p < 30) return Math.round(80 + ((p - 20) / 10) * 3);
        if (p < 50) return Math.round(83 + ((p - 30) / 20) * 9);
        return Math.round(92 + ((p - 50) / 50) * 5);
    };

    // ── Format date ───────────────────────────────────────────────────────────
    const formatDate = (isoString) => {
        if (!isoString) return '—';
        try {
            return new Date(isoString).toLocaleString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
            });
        } catch {
            return isoString;
        }
    };

    // ── Export single record to PDF using shared exportToPdf utility ──────────
    const handleExportPdf = async (record, rankIndex = 0) => {
        setExportingId(record.id);
        try {
            const topThree = (record.top_predictions ?? []).slice(0, 3);
            const selectedCareer = topThree[rankIndex]?.career_path || record.prediction_result;
            const roadmapByPath = record.learning_roadmap_by_path ?? null;
            const certsByPath = record.certification_data_by_path ?? null;
            const keywordsByPath = record.extracted_keywords_by_path ?? null;

            await exportToPdf({
                topThree,
                calculateProfileFit,
                learningRoadmap: roadmapByPath?.[selectedCareer] ?? record.learning_roadmap ?? null,
                certificationData: certsByPath?.[selectedCareer] ?? record.certification_data ?? null,
                careerContent: null,
                logoRef: null,
                extractedKeywords: keywordsByPath?.[selectedCareer] ?? (record.extracted_keywords ?? []),
                selectedCareerPath: selectedCareer,
            });
        } catch (err) {
            console.error('PDF export failed:', err);
        } finally {
            setExportingId(null);
            setExportMenuOpenId(null);
        }
    };

    const getResumeHash = (text) => {
        if (!text) return null;
        return `${text.substring(0, 100)}_${text.length}`;
    };

    const handleViewDetailedAnalysis = (record) => {
        const resumeText = record.input_data || '';
        const rawConfidence = parseFloat(record.confidence_score || 0);
        const extractedKeywords = Array.isArray(record.extracted_keywords) ? record.extracted_keywords : [];
        const extractedKeywordsByPath = record.extracted_keywords_by_path || {};
        const totalDistinctiveKeywordsByPath = record.total_distinctive_keywords_by_path || {};
        const totalDistinctiveKeywords = typeof record.total_distinctive_keywords === 'number'
            ? record.total_distinctive_keywords
            : extractedKeywords.length;
        const topPredictions = Array.isArray(record.top_predictions) && record.top_predictions.length > 0
            ? record.top_predictions
            : [{ career_path: record.prediction_result, raw_confidence: rawConfidence }];

        setPredictionResults({
            prediction: record.prediction_result,
            raw_confidence: rawConfidence,
            top_predictions: topPredictions,
            extracted_keywords: extractedKeywords,
            extracted_keywords_by_path: extractedKeywordsByPath,
            total_distinctive_keywords: totalDistinctiveKeywords,
            total_distinctive_keywords_by_path: totalDistinctiveKeywordsByPath,
        });

        setResumeText(resumeText);

        const resumeHash = getResumeHash(resumeText);
        const selectedCareer = topPredictions[0]?.career_path || record.prediction_result;
        const hydrateMap = (map) => {
            if (!map) return null;
            return Object.fromEntries(
                Object.entries(map).map(([career, data]) => [
                    career,
                    data && !data.resumeHash ? { ...data, resumeHash } : data,
                ])
            );
        };

        if (record.learning_roadmap_by_path) {
            setLearningRoadmapByPath(hydrateMap(record.learning_roadmap_by_path));
        } else if (record.learning_roadmap) {
            setLearningRoadmapByPath({
                [selectedCareer]: { ...record.learning_roadmap, resumeHash },
            });
        } else {
            setLearningRoadmapByPath(null);
        }

        if (record.certification_data_by_path) {
            setCertificationDataByPath(hydrateMap(record.certification_data_by_path));
        } else if (record.certification_data) {
            setCertificationDataByPath({
                [selectedCareer]: { ...record.certification_data, resumeHash },
            });
        } else {
            setCertificationDataByPath(null);
        }

        if (record.skills_insights_by_path) {
            setSkillsInsightsByPath(record.skills_insights_by_path);
        } else {
            setSkillsInsightsByPath(null);
        }

        setHistoryRecordId(record.id || null);
        navigate(`/learnmore?career=${encodeURIComponent(selectedCareer)}`);
    };

    const handleViewDetailedAnalysisForRank = (record, rankIndex = 0) => {
        const resumeText = record.input_data || '';
        const rawConfidence = parseFloat(record.confidence_score || 0);
        const extractedKeywords = Array.isArray(record.extracted_keywords) ? record.extracted_keywords : [];
        const extractedKeywordsByPath = record.extracted_keywords_by_path || {};
        const totalDistinctiveKeywordsByPath = record.total_distinctive_keywords_by_path || {};
        const totalDistinctiveKeywords = typeof record.total_distinctive_keywords === 'number'
            ? record.total_distinctive_keywords
            : extractedKeywords.length;
        const topPredictions = Array.isArray(record.top_predictions) && record.top_predictions.length > 0
            ? record.top_predictions
            : [{ career_path: record.prediction_result, raw_confidence: rawConfidence }];
        const selectedCareer = topPredictions[rankIndex]?.career_path || record.prediction_result;

        setPredictionResults({
            prediction: record.prediction_result,
            raw_confidence: rawConfidence,
            top_predictions: topPredictions,
            extracted_keywords: extractedKeywords,
            extracted_keywords_by_path: extractedKeywordsByPath,
            total_distinctive_keywords: totalDistinctiveKeywords,
            total_distinctive_keywords_by_path: totalDistinctiveKeywordsByPath,
        });

        setResumeText(resumeText);

        const resumeHash = getResumeHash(resumeText);
        const hydrateMap = (map) => {
            if (!map) return null;
            return Object.fromEntries(
                Object.entries(map).map(([career, data]) => [
                    career,
                    data && !data.resumeHash ? { ...data, resumeHash } : data,
                ])
            );
        };

        if (record.learning_roadmap_by_path) {
            setLearningRoadmapByPath(hydrateMap(record.learning_roadmap_by_path));
        } else if (record.learning_roadmap) {
            setLearningRoadmapByPath({
                [selectedCareer]: { ...record.learning_roadmap, resumeHash },
            });
        } else {
            setLearningRoadmapByPath(null);
        }

        if (record.certification_data_by_path) {
            setCertificationDataByPath(hydrateMap(record.certification_data_by_path));
        } else if (record.certification_data) {
            setCertificationDataByPath({
                [selectedCareer]: { ...record.certification_data, resumeHash },
            });
        } else {
            setCertificationDataByPath(null);
        }

        if (record.skills_insights_by_path) {
            setSkillsInsightsByPath(record.skills_insights_by_path);
        } else {
            setSkillsInsightsByPath(null);
        }

        setHistoryRecordId(record.id || null);
        setAnalysisMenuOpenId(null);
        navigate(`/learnmore?career=${encodeURIComponent(selectedCareer)}`);
    };

    // ── Render ────────────────────────────────────────────────────────────────
    return (
        <div className="history-container">
            {/* Header */}
            <header className="history-header">
                <div className="history-header-content">
                    <div className="history-header-left">
                        <h1 className="history-brand" onClick={() => navigate('/')}>
                            <Logo variant="modern" />
                        </h1>
                    </div>

                    <nav className="history-top-nav">
                        <button className="history-nav-tab" onClick={() => navigate('/dashboard')}>Dashboard</button>
                        <button className="history-nav-tab" onClick={() => setShowCareerPaths(true)}>Career Paths</button>
                        <button className="history-nav-tab active" type="button" disabled>History</button>
                    </nav>

                    <div className="history-header-right">
                        {currentUser && (
                            <div className="history-profile-chip">
                                <span className="history-profile-dot" aria-hidden="true"></span>
                                <span className="history-greeting">{currentUser.username}</span>
                            </div>
                        )}
                        <div className="history-action-group">
                            {currentUser?.is_admin && (
                                <button className="history-admin-btn" onClick={() => navigate('/admin')}>
                                    🛡 Admin
                                </button>
                            )}
                            <button className="history-logout-btn" onClick={() => setShowLogoutConfirm(true)}>
                                Logout
                            </button>
                        </div>
                    </div>

                    {/* Hamburger Button (mobile only) */}
                    <button
                        className={`history-hamburger ${menuOpen ? 'open' : ''}`}
                        onClick={() => setMenuOpen((prev) => !prev)}
                        aria-label="Toggle navigation menu"
                        aria-expanded={menuOpen}
                    >
                        <span className="hamburger-line" />
                        <span className="hamburger-line" />
                        <span className="hamburger-line" />
                    </button>
                </div>

                {showLogoutConfirm && (
                    <div className="history-logout-bar">
                        <span>Are you sure you want to logout?</span>
                        <div className="history-logout-actions">
                            <button className="logout-yes-btn" onClick={handleLogout}>Yes, Logout</button>
                            <button className="logout-cancel-btn" onClick={() => setShowLogoutConfirm(false)}>Cancel</button>
                        </div>
                    </div>
                )}
            </header>

            {/* Mobile Dropdown Menu */}
            {menuOpen && (
                <div className="history-mobile-menu" ref={menuRef}>
                    {currentUser && (
                        <div className="mobile-menu-greeting">
                            Hello, <strong>{currentUser.username}</strong>
                        </div>
                    )}
                    <button
                        className="mobile-menu-item dashboard-item"
                        onClick={() => { setMenuOpen(false); setShowCareerPaths(true); }}
                    >
                        Career Paths
                    </button>
                    <button
                        className="mobile-menu-item dashboard-item"
                        type="button"
                        disabled
                    >
                        History
                    </button>
                    <button
                        className="mobile-menu-item dashboard-item"
                        onClick={() => { setMenuOpen(false); navigate('/dashboard'); }}
                    >
                        Dashboard
                    </button>
                    {currentUser?.is_admin && (
                        <button
                            className="mobile-menu-item dashboard-item"
                            onClick={() => { setMenuOpen(false); navigate('/admin'); }}
                        >
                            🛡 Admin
                        </button>
                    )}
                    <button
                        className="mobile-menu-item logout-item"
                        onClick={() => { setMenuOpen(false); setShowLogoutConfirm(true); }}
                    >
                        Logout
                    </button>
                </div>
            )}

            {/* Main */}
            <main className="history-main">
                <div className="history-content">
                    <div className="history-page-header">
                        <div className="history-title-area">
                            <h2 className="history-page-title">
                                Prediction History
                            </h2>
                            <p className="history-page-subtitle">
                                All your past career path predictions, newest first.
                            </p>
                        </div>
                    </div>

                    {/* States */}
                    {isLoading ? (
                        <div className="history-loading">
                            <div className="history-spinner" />
                            <p>Loading your prediction history...</p>
                        </div>
                    ) : error ? (
                        <div className="history-error">
                            <span className="history-error-icon">
                                {React.createElement(otherIcons['FaExclamationTriangle'], { size: 18, color: '#ef4444' })}
                            </span>
                            <span>{error}</span>
                        </div>
                    ) : history.length === 0 ? (
                        <div className="history-empty">
                            <div className="history-empty-icon">
                                {React.createElement(otherIcons['FaClipboardList'], { size: 56, color: '#cbd5e1' })}
                            </div>
                            <h3>No Predictions Yet</h3>
                            <p>Run your first career path prediction to see your history here.</p>
                            <button className="history-go-dashboard-btn" onClick={() => navigate('/dashboard')}>
                                Go to Dashboard
                            </button>
                        </div>
                    ) : (
                        <>
                            {/* Desktop / Tablet Table */}
                            <div className="history-table-wrapper">
                                <table className="history-table">
                                    <thead>
                                        <tr>
                                            <th>#</th>
                                            <th>Date &amp; Time</th>
                                            <th>Predicted Career</th>
                                            <th>Resume File</th>
                                            <th>Profile Fit</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {paginatedHistory.map((record, idx) => {
                                            const topThree = (record.top_predictions ?? []).slice(0, 3);
                                            return (
                                            <tr key={record.id} className="history-row">
                                                <td className="history-rank">{startIndex + idx + 1}</td>
                                                <td className="history-date">{formatDate(record.date_created)}</td>
                                                <td className="history-career">
                                                    <span className="career-badge">{record.prediction_result}</span>
                                                </td>
                                                <td className="history-file">
                                                    {record.filename
                                                        ? <span className="filename-chip" title={record.filename}>
                                                            {React.createElement(otherIcons['FaFile'], { size: 12, color: '#64748b' })}
                                                            {record.filename}
                                                        </span>
                                                        : <span className="no-data">—</span>
                                                    }
                                                </td>
                                                <td className="history-confidence">
                                                    {record.confidence_score != null ? (
                                                        <div className="confidence-wrapper">
                                                            <span className="confidence-value">
                                                                {calculateProfileFit(parseFloat(record.confidence_score))}%
                                                            </span>
                                                            <div className="confidence-bar-track">
                                                                <div
                                                                    className="confidence-bar-fill"
                                                                    style={{ width: `${calculateProfileFit(parseFloat(record.confidence_score))}%` }}
                                                                />
                                                            </div>
                                                        </div>
                                                    ) : (
                                                        <span className="no-data">—</span>
                                                    )}
                                                </td>
                                                <td className="history-actions">
                                                    <div className="export-menu" onClick={(e) => e.stopPropagation()}>
                                                        <button
                                                            className={`pdf-download-btn ${exportingId === record.id ? 'loading' : ''}`}
                                                            onClick={() => handleExportPdf(record, 0)}
                                                            disabled={exportingId === record.id}
                                                            title="Download PDF Report"
                                                        >
                                                            {exportingId === record.id ? (
                                                                <>
                                                                    {React.createElement(otherIcons['FaSpinner'], { size: 13 })}
                                                                    <span>Generating...</span>
                                                                </>
                                                            ) : (
                                                                <>
                                                                    {React.createElement(otherIcons['FaDownload'], { size: 13 })}
                                                                    <span>Download PDF</span>
                                                                </>
                                                            )}
                                                        </button>
                                                        <button
                                                            className="export-menu-toggle"
                                                            onClick={() => setExportMenuOpenId((prev) => prev === record.id ? null : record.id)}
                                                            title="Choose which match to export"
                                                            aria-expanded={exportMenuOpenId === record.id}
                                                        >
                                                            {React.createElement(otherIcons['FaChevronDown'], { size: 12 })}
                                                        </button>
                                                        {exportMenuOpenId === record.id && (
                                                            <div className="export-menu-dropdown">
                                                                <button onClick={() => handleExportPdf(record, 0)}>
                                                                    Export #1 • {topThree[0]?.career_path || 'Primary Match'}
                                                                </button>
                                                                {topThree[1] && (
                                                                    <button onClick={() => handleExportPdf(record, 1)}>
                                                                        Export #2 • {topThree[1].career_path}
                                                                    </button>
                                                                )}
                                                                {topThree[2] && (
                                                                    <button onClick={() => handleExportPdf(record, 2)}>
                                                                        Export #3 • {topThree[2].career_path}
                                                                    </button>
                                                                )}
                                                            </div>
                                                        )}
                                                    </div>
                                                    <div className="analysis-menu" onClick={(e) => e.stopPropagation()}>
                                                        <button
                                                            className="view-analysis-btn"
                                                            onClick={() => handleViewDetailedAnalysisForRank(record, 0)}
                                                            title="View Detailed Analysis"
                                                        >
                                                            {React.createElement(otherIcons['FaArrowRight'], { size: 12 })}
                                                            <span>View Detailed Analysis</span>
                                                        </button>
                                                        <button
                                                            className="analysis-menu-toggle"
                                                            onClick={() => setAnalysisMenuOpenId((prev) => prev === record.id ? null : record.id)}
                                                            aria-expanded={analysisMenuOpenId === record.id}
                                                            title="Choose which match to view"
                                                        >
                                                            {React.createElement(otherIcons['FaChevronDown'], { size: 12 })}
                                                        </button>
                                                        {analysisMenuOpenId === record.id && (
                                                            <div className="analysis-menu-dropdown">
                                                                <button onClick={() => handleViewDetailedAnalysisForRank(record, 0)}>
                                                                    View #1 • {topThree[0]?.career_path || 'Primary Match'}
                                                                </button>
                                                                {topThree[1] && (
                                                                    <button onClick={() => handleViewDetailedAnalysisForRank(record, 1)}>
                                                                        View #2 • {topThree[1].career_path}
                                                                    </button>
                                                                )}
                                                                {topThree[2] && (
                                                                    <button onClick={() => handleViewDetailedAnalysisForRank(record, 2)}>
                                                                        View #3 • {topThree[2].career_path}
                                                                    </button>
                                                                )}
                                                            </div>
                                                        )}
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                        })}
                                    </tbody>
                                </table>
                            </div>

                            {/* Mobile Cards (hidden on desktop via CSS) */}
                            <div className="history-cards">
                                {paginatedHistory.map((record, idx) => {
                                    const topThree = (record.top_predictions ?? []).slice(0, 3);
                                    return (
                                    <div key={record.id} className="history-card">
                                        <div className="history-card-top">
                                            <span className="history-card-rank">#{startIndex + idx + 1}</span>
                                            <span className="career-badge">{record.prediction_result}</span>
                                        </div>
                                        <div className="history-card-meta">
                                            <span className="history-card-label">Date</span>
                                            <span className="history-card-value">{formatDate(record.date_created)}</span>
                                        </div>
                                        {record.filename && (
                                            <div className="history-card-meta">
                                                <span className="history-card-label">File</span>
                                                <span className="filename-chip" title={record.filename}>
                                                    {React.createElement(otherIcons['FaFile'], { size: 11, color: '#64748b' })}
                                                    {record.filename}
                                                </span>
                                            </div>
                                        )}
                                        {record.confidence_score != null && (
                                            <div className="history-card-meta">
                                                <span className="history-card-label">Profile Fit</span>
                                                <div className="confidence-wrapper" style={{ flex: 1 }}>
                                                    <span className="confidence-value">
                                                        {calculateProfileFit(parseFloat(record.confidence_score))}%
                                                    </span>
                                                    <div className="confidence-bar-track">
                                                        <div
                                                            className="confidence-bar-fill"
                                                            style={{ width: `${calculateProfileFit(parseFloat(record.confidence_score))}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                        <div className="history-card-actions">
                                            <div className="export-menu" onClick={(e) => e.stopPropagation()}>
                                                <button
                                                    className={`pdf-download-btn ${exportingId === record.id ? 'loading' : ''}`}
                                                    onClick={() => handleExportPdf(record, 0)}
                                                    disabled={exportingId === record.id}
                                                >
                                                    {exportingId === record.id ? (
                                                        <>
                                                            {React.createElement(otherIcons['FaSpinner'], { size: 13 })}
                                                            <span>Generating...</span>
                                                        </>
                                                    ) : (
                                                        <>
                                                            {React.createElement(otherIcons['FaDownload'], { size: 13 })}
                                                            <span>Download PDF</span>
                                                        </>
                                                    )}
                                                </button>
                                                <button
                                                    className="export-menu-toggle"
                                                    onClick={() => setExportMenuOpenId((prev) => prev === record.id ? null : record.id)}
                                                    aria-expanded={exportMenuOpenId === record.id}
                                                >
                                                    {React.createElement(otherIcons['FaChevronDown'], { size: 12 })}
                                                </button>
                                                {exportMenuOpenId === record.id && (
                                                    <div className="export-menu-dropdown">
                                                        <button onClick={() => handleExportPdf(record, 0)}>
                                                            Export #1 • {topThree[0]?.career_path || 'Primary Match'}
                                                        </button>
                                                        {topThree[1] && (
                                                            <button onClick={() => handleExportPdf(record, 1)}>
                                                                Export #2 • {topThree[1].career_path}
                                                            </button>
                                                        )}
                                                        {topThree[2] && (
                                                            <button onClick={() => handleExportPdf(record, 2)}>
                                                                Export #3 • {topThree[2].career_path}
                                                            </button>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                            <div className="analysis-menu" onClick={(e) => e.stopPropagation()}>
                                                <button
                                                    className="view-analysis-btn"
                                                    onClick={() => handleViewDetailedAnalysisForRank(record, 0)}
                                                >
                                                    {React.createElement(otherIcons['FaArrowRight'], { size: 12 })}
                                                    <span>View Detailed Analysis</span>
                                                </button>
                                                <button
                                                    className="analysis-menu-toggle"
                                                    onClick={() => setAnalysisMenuOpenId((prev) => prev === record.id ? null : record.id)}
                                                    aria-expanded={analysisMenuOpenId === record.id}
                                                >
                                                    {React.createElement(otherIcons['FaChevronDown'], { size: 12 })}
                                                </button>
                                                {analysisMenuOpenId === record.id && (
                                                    <div className="analysis-menu-dropdown">
                                                        <button onClick={() => handleViewDetailedAnalysisForRank(record, 0)}>
                                                            View #1 • {topThree[0]?.career_path || 'Primary Match'}
                                                        </button>
                                                        {topThree[1] && (
                                                            <button onClick={() => handleViewDetailedAnalysisForRank(record, 1)}>
                                                                View #2 • {topThree[1].career_path}
                                                            </button>
                                                        )}
                                                        {topThree[2] && (
                                                            <button onClick={() => handleViewDetailedAnalysisForRank(record, 2)}>
                                                                View #3 • {topThree[2].career_path}
                                                            </button>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                );
                                })}
                            </div>

                            {history.length > RECORDS_PER_PAGE && (
                                <div className="history-pagination">
                                    <button
                                        className="history-page-btn"
                                        onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
                                        disabled={currentPage === 1}
                                    >
                                        Previous
                                    </button>
                                    <span className="history-page-indicator">
                                        Page {currentPage} of {totalPages}
                                    </span>
                                    <button
                                        className="history-page-btn"
                                        onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
                                        disabled={currentPage === totalPages}
                                    >
                                        Next
                                    </button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </main>

            {/* Career Paths Modal */}
            {showCareerPaths && (
                <CareerPathsModal onClose={() => setShowCareerPaths(false)} />
            )}
        </div>
    );
};

export default History;
