import React, { useState, useEffect, useRef } from 'react';
import { FaExternalLinkAlt, FaSpinner } from 'react-icons/fa';
import { otherIcons } from '../../../../utils/otherIcons';
import apiService from '../../../../services/api/apiService';
import './CertificationPanel.css';

/**
 * CertificationPanel - Displays AI-generated personalized certification recommendations
 * 
 * @param {Object} props
 * @param {string} props.careerPath - The matched career path name
 * @param {Array} props.growthAreas - Suggested skills to develop (fallback)
 * @param {string} props.resumeText - Raw resume text for Gemini analysis
 * @param {Object} props.certificationData - Cached certifications from context
 * @param {Function} props.setCertificationData - Function to update cached certifications
 */
const CertificationPanel = ({
    careerPath,
    growthAreas = [],
    resumeText,
    certificationData,
    setCertificationData
}) => {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const hasFetchedRef = useRef(false);

    useEffect(() => {
        // Generate a simple hash from resumeText to identify unique resumes
        const getResumeHash = (text) => {
            if (!text) return null;
            // Simple hash: first 100 chars + length (good enough for cache validation)
            return `${text.substring(0, 100)}_${text.length}`;
        };

        const currentResumeHash = getResumeHash(resumeText);

        // If we have cached certifications, check if it's for the current resume
        if (certificationData) {
            // If resume has changed (different hash), invalidate cache
            if (certificationData.resumeHash !== currentResumeHash) {
                setCertificationData(null); // Clear cache and trigger regeneration
                return;
            }
            // Cache is valid for this resume, don't fetch again
            return;
        }

        // If no resume text available, can't generate certifications
        if (!resumeText) {
            setError('Resume text not available. Please upload and analyze a resume first.');
            return;
        }

        // Prevent double fetch in React Strict Mode
        if (hasFetchedRef.current) {
            return;
        }

        // Fetch certifications from API
        const fetchCertifications = async () => {
            hasFetchedRef.current = true;
            setIsLoading(true);
            setError(null);

            try {
                const response = await apiService.generateCertifications(careerPath, resumeText);

                if (response.success && response.certifications) {
                    // Attach resume hash to the certifications for cache validation
                    const certificationsWithHash = {
                        ...response.certifications,
                        resumeHash: currentResumeHash
                    };
                    setCertificationData(certificationsWithHash);
                } else {
                    throw new Error('Invalid response from server');
                }
            } catch (err) {
                console.error('Error fetching certifications:', err);
                setError(err.message || 'Failed to generate certifications. Please try again.');
                hasFetchedRef.current = false; // Reset on error so retry works
            } finally {
                setIsLoading(false);
            }
        };

        fetchCertifications();
    }, [careerPath, resumeText, certificationData, setCertificationData]);

    // Helper to get level badge class
    const getLevelClass = (level) => {
        switch (level) {
            case 'beginner': return 'level-beginner';
            case 'intermediate': return 'level-intermediate';
            case 'advanced': return 'level-advanced';
            default: return 'level-intermediate';
        }
    };

    // Loading state
    if (isLoading) {
        return (
            <div className="certification-panel">
                <div className="certification-loading">
                    <FaSpinner className="certification-spinner" />
                    <p>Finding your certifications...</p>
                </div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className="certification-panel">
                <div className="certification-error">
                    <div className="error-icon">⚠️</div>
                    <h3>Failed to Generate Certifications</h3>
                    <p className="error-message">{error}</p>
                    {resumeText && (
                        <button className="retry-button" onClick={() => window.location.reload()}>
                            Try Again
                        </button>
                    )}
                </div>
            </div>
        );
    }

    // Helper to sort certifications by level
    const sortCertificationsByLevel = (certs) => {
        const levelOrder = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3
        };

        return [...certs].sort((a, b) => {
            const levelA = levelOrder[a.level.toLowerCase()] || 99;
            const levelB = levelOrder[b.level.toLowerCase()] || 99;
            return levelA - levelB;
        });
    };

    // Content state
    if (certificationData) {
        const sortedCertifications = sortCertificationsByLevel(certificationData.certifications);

        return (
            <div className="certification-panel">
                <div className="certification-header">
                    <div className="certification-header-content">
                        <div>
                            <h3>Recommended Certifications</h3>
                            <p className="certification-subtitle">{certificationData.summary}</p>
                        </div>
                    </div>
                </div>

                <div className="certifications-container">
                    {sortedCertifications.map((cert, index) => (
                        <div key={cert.id} className="cert-card" style={{ animationDelay: `${index * 100}ms` }}>
                            <div className="cert-header">
                                <h4 className="cert-name">{cert.name}</h4>
                                <span className={`cert-level ${getLevelClass(cert.level)}`}>
                                    {cert.level}
                                </span>
                            </div>
                            <div className="cert-provider">{cert.provider}</div>
                            <p className="cert-why">{cert.why}</p>
                            <div className="cert-footer">
                                <span className="cert-duration">
                                    {React.createElement(otherIcons["FaCalendarCheck"], { style: { color: '#2563eb', marginRight: '2px', size: 32 } })}
                                    {cert.estimated_duration}
                                </span>
                                <a
                                    href={cert.search_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="cert-link"
                                >
                                    View Certification <FaExternalLinkAlt size={12} />
                                </a>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    // Fallback if no data
    return (
        <div className="certification-panel">
            <div className="certification-coming-soon">
                <div className="coming-soon-icon">
                    {React.createElement(otherIcons["GrCertificate"])}
                </div>
                <h3 className="coming-soon-title">No Certifications Available</h3>
                <p className="coming-soon-description">
                    Upload and analyze a resume to see your personalized certification recommendations.
                </p>
            </div>
        </div>
    );
};

export default CertificationPanel;
