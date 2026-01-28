import React from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../../components/common/Logo';
import './HowItWorks.css';

const HowItWorks = () => {
    const navigate = useNavigate();

    return (
        <div className="hiw-container">
            {/* Header */}
            <header className="hiw-header">
                <h1 className="hiw-brand" onClick={() => navigate('/')}>
                    <Logo variant="modern" />
                </h1>
                <button className="hiw-back-btn" onClick={() => navigate('/dashboard')}>
                    ← Back to Dashboard
                </button>
            </header>

            {/* Main Content */}
            <main className="hiw-main">
                <h2 className="hiw-title">How We Calculate Your Career Matches</h2>
                <p className="hiw-subtitle">
                    Understanding the AI behind your personalized career recommendations
                </p>

                {/* Placeholder Box - Fills remaining space */}
                <div className="hiw-placeholder">
                    <div className="hiw-placeholder-icon">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="#2563eb">
                            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z" />
                        </svg>
                    </div>
                    <h3>Coming Soon</h3>
                    <p>Detailed explanation of our AI prediction methodology will be available here.</p>
                </div>
            </main>
        </div>
    );
};

export default HowItWorks;
