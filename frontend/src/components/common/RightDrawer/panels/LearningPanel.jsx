import React from 'react';
import { FaGraduationCap } from 'react-icons/fa';
import './LearningPanel.css';

/**
 * LearningPanel - Displays learning resources for skill development
 * 
 * @param {Object} props
 * @param {string} props.careerPath - The matched career path name
 * @param {Array} props.growthAreas - Suggested skills to develop
 */
const LearningPanel = ({ careerPath, growthAreas = [] }) => {
    return (
        <div className="learning-panel">
            <div className="learning-coming-soon">
                <div className="coming-soon-icon">
                    <FaGraduationCap />
                </div>
                <h3 className="coming-soon-title">Coming Soon</h3>
                <p className="coming-soon-description">
                    Curated courses and learning resources for{' '}
                    <strong>{careerPath || 'your career path'}</strong> are being prepared.
                </p>
                <p className="coming-soon-note">
                    We're gathering the best learning materials to help you grow in this field. Check back soon!
                </p>
            </div>
        </div>
    );
};

export default LearningPanel;
