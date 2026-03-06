import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { exportToPdf } from '../../utils/exportToPdf';
import Logo from '../../components/common/Logo';
import apiService from '../../services/api/apiService';
import { useAuth } from '../../context/AuthContext';
import { otherIcons } from '../../utils/otherIcons';
import './History.css';

const History = () => {
    const navigate = useNavigate();
    const { currentUser, logout, getToken } = useAuth();
    const [history, setHistory] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
    const [exportingId, setExportingId] = useState(null);
    const [menuOpen, setMenuOpen] = useState(false);
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

    // ── Profile Fit (mirrors Dashboard logic exactly) ──────────────────────
    const calculateProfileFit = (rawProbability) => {
        const p = rawProbability;
        if (p < 5) return Math.round((p / 5) * 35);
        if (p < 10) return Math.round(35 + ((p - 5) / 5) * 15);
        if (p < 15) return Math.round(50 + ((p - 10) / 5) * 15);
        if (p < 20) return Math.round(65 + ((p - 15) / 5) * 18);
        if (p < 30) return Math.round(83 + ((p - 20) / 10) * 2);
        if (p < 50) return Math.round(85 + ((p - 30) / 20) * 5);
        return Math.round(90 + ((p - 50) / 50) * 5);
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
    const handleExportPdf = async (record) => {
        setExportingId(record.id);
        try {
            await exportToPdf({
                topThree: (record.top_predictions ?? []).slice(0, 3),
                calculateProfileFit,
                learningRoadmap: record.learning_roadmap ?? null,
                certificationData: record.certification_data ?? null,
                careerContent: null,
                logoRef: null,
            });
        } catch (err) {
            console.error('PDF export failed:', err);
        } finally {
            setExportingId(null);
        }
    };

    // ── Render ────────────────────────────────────────────────────────────────
    return (
        <div className="history-container">
            {/* Header */}
            <header className="history-header">
                <div className="history-header-content">
                    <h1 className="history-brand" onClick={() => navigate('/')}>
                        <Logo variant="modern" />
                    </h1>

                    {/* Desktop Nav */}
                    <div className="history-header-actions">
                        {currentUser && (
                            <span className="history-greeting">Hello, {currentUser.username}</span>
                        )}
                        <button className="history-dashboard-btn" onClick={() => navigate('/dashboard')}>
                            Dashboard
                        </button>
                        <button className="history-logout-btn" onClick={() => setShowLogoutConfirm(true)}>
                            Logout
                        </button>
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
                        onClick={() => { setMenuOpen(false); navigate('/dashboard'); }}
                    >
                        Dashboard
                    </button>
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
                                        {history.map((record, idx) => (
                                            <tr key={record.id} className="history-row">
                                                <td className="history-rank">{idx + 1}</td>
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
                                                    <button
                                                        className={`pdf-download-btn ${exportingId === record.id ? 'loading' : ''}`}
                                                        onClick={() => handleExportPdf(record)}
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
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            {/* Mobile Cards (hidden on desktop via CSS) */}
                            <div className="history-cards">
                                {history.map((record, idx) => (
                                    <div key={record.id} className="history-card">
                                        <div className="history-card-top">
                                            <span className="history-card-rank">#{idx + 1}</span>
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
                                            <button
                                                className={`pdf-download-btn ${exportingId === record.id ? 'loading' : ''}`}
                                                onClick={() => handleExportPdf(record)}
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
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}
                </div>
            </main>
        </div>
    );
};

export default History;
