import React from 'react';
import { FaBriefcase, FaBook, FaCertificate } from 'react-icons/fa';
import './RightDock.css';

/**
 * RightDock - A floating vertical button dock on the right edge of the screen
 * 
 * @param {Object} props
 * @param {string|null} props.activePanel - Currently active panel ('jobs' | 'learning' | 'certifications' | null)
 * @param {Function} props.onPanelToggle - Callback when a panel button is clicked
 */
const RightDock = ({ activePanel, onPanelToggle }) => {
    const handleJobsClick = () => {
        onPanelToggle(activePanel === 'jobs' ? null : 'jobs');
    };

    const handleLearningClick = () => {
        onPanelToggle(activePanel === 'learning' ? null : 'learning');
    };

    const handleCertificationsClick = () => {
        onPanelToggle(activePanel === 'certifications' ? null : 'certifications');
    };

    return (
        <div className="right-dock" role="toolbar" aria-label="Quick access panel">
            <button
                className={`dock-button ${activePanel === 'jobs' ? 'dock-button--active' : ''}`}
                onClick={handleJobsClick}
                aria-expanded={activePanel === 'jobs'}
                aria-controls="jobs-drawer"
                aria-label="Open Jobs panel"
                title="Jobs"
            >
                <FaBriefcase className="dock-button-icon" />
                <span className="dock-button-label">Jobs</span>
            </button>

            <button
                className={`dock-button ${activePanel === 'learning' ? 'dock-button--active' : ''}`}
                onClick={handleLearningClick}
                aria-expanded={activePanel === 'learning'}
                aria-controls="learning-drawer"
                aria-label="Open Learning Materials panel"
                title="Learning"
            >
                <FaBook className="dock-button-icon" />
                <span className="dock-button-label">Learning</span>
            </button>

            <button
                className={`dock-button ${activePanel === 'certifications' ? 'dock-button--active' : ''}`}
                onClick={handleCertificationsClick}
                aria-expanded={activePanel === 'certifications'}
                aria-controls="certifications-drawer"
                aria-label="Open Certifications panel"
                title="Certifications"
            >
                <FaCertificate className="dock-button-icon" />
                <span className="dock-button-label">Certifications</span>
            </button>
        </div>
    );
};

export default RightDock;
