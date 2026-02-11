import React from 'react';
import { FaCertificate } from 'react-icons/fa';
import './CertificationPanel.css';

/**
 * CertificationPanel - Displays certification opportunities for career advancement
 * 
 * @param {Object} props
 * @param {string} props.careerPath - The matched career path name
 * @param {Array} props.growthAreas - Suggested skills/areas for certification
 */
const CertificationPanel = ({ careerPath, growthAreas = [] }) => {
    return (
        <div className="certification-panel">
            <div className="certification-coming-soon">
                <div className="coming-soon-icon">
                    <FaCertificate />
                </div>
                <h3 className="coming-soon-title">Coming Soon</h3>
                <p className="coming-soon-description">
                    Industry-recognized certifications for{' '}
                    <strong>{careerPath || 'your career path'}</strong> are being curated.
                </p>
                <p className="coming-soon-note">
                    We're compiling the most valuable certifications to advance your career. Check back soon!
                </p>
            </div>
        </div>
    );
};

export default CertificationPanel;
