import React, { useState, useEffect, useRef } from 'react';
import { FaGraduationCap, FaYoutube, FaBook, FaFileAlt } from 'react-icons/fa';
import apiService from '../../../../services/api/apiService';
import './LearningPanel.css';

/**
 * LearningPanel - Displays AI-generated personalized learning roadmap
 * 
 * @param {Object} props
 * @param {string} props.careerPath - The matched career path name
 * @param {Array} props.growthAreas - Suggested skills to develop (fallback)
 * @param {string} props.resumeText - Raw resume text for Gemini analysis
 * @param {Object} props.learningRoadmap - Cached roadmap from context
 * @param {Function} props.setLearningRoadmap - Function to update cached roadmap
 */
const LearningPanel = ({
    careerPath,
    growthAreas = [],
    resumeText,
    learningRoadmap,
    setLearningRoadmap
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

        // If we have a cached roadmap, check if it's for the current resume
        if (learningRoadmap) {
            // If resume has changed (different hash), invalidate cache
            if (learningRoadmap.resumeHash !== currentResumeHash) {
                setLearningRoadmap(null); // Clear cache and trigger regeneration
                return;
            }
            // Cache is valid for this resume, don't fetch again
            return;
        }

        // If no resume text available, can't generate roadmap
        if (!resumeText) {
            setError('Resume text not available. Please upload and analyze a resume first.');
            return;
        }

        // Prevent double fetch in React Strict Mode
        if (hasFetchedRef.current) {
            return;
        }

        // Fetch roadmap from API
        const fetchRoadmap = async () => {
            hasFetchedRef.current = true;
            setIsLoading(true);
            setError(null);

            try {
                const response = await apiService.generateLearningRoadmap(careerPath, resumeText);

                if (response.success && response.roadmap) {
                    // Attach resume hash to the roadmap for cache validation
                    const roadmapWithHash = {
                        ...response.roadmap,
                        resumeHash: currentResumeHash
                    };
                    setLearningRoadmap(roadmapWithHash);
                } else {
                    throw new Error('Invalid response from server');
                }
            } catch (err) {
                console.error('Error fetching learning roadmap:', err);
                setError(err.message || 'Failed to generate learning roadmap. Please try again.');
                hasFetchedRef.current = false; // Reset on error so retry works
            } finally {
                setIsLoading(false);
            }
        };

        fetchRoadmap();
    }, [careerPath, resumeText, learningRoadmap, setLearningRoadmap]);

    // Helper to get icon for resource type
    const getResourceIcon = (type) => {
        switch (type) {
            case 'video': return <FaYoutube />;
            case 'course': return <FaBook />;
            case 'article': return <FaFileAlt />;
            default: return <FaFileAlt />;
        }
    };

    // Loading state
    if (isLoading) {
        return (
            <div className="learning-panel">
                <div className="learning-loading">
                    <div className="loading-spinner"></div>
                    <h3>Generating Your Personalized Roadmap...</h3>
                    <p>Analyzing your resume and creating a tailored learning path</p>
                </div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className="learning-panel">
                <div className="learning-error">
                    <div className="error-icon">⚠️</div>
                    <h3>Failed to Generate Roadmap</h3>
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

    // Content state
    if (learningRoadmap) {
        return (
            <div className="learning-panel">
                <div className="roadmap-header">
                    <div className="roadmap-header-content">
                        <div>
                            <h3>Resume Analysis</h3>
                            <p className="roadmap-subtitle">{learningRoadmap.analysis_summary}</p>
                        </div>
                    </div>
                </div>

                <div className="improvement-areas-container">
                    {learningRoadmap.improvement_areas.map((area, index) => (
                        <div key={area.id} className="skill-card" style={{ animationDelay: `${index * 100}ms` }}>
                            <div className="skill-header">
                                <div className="skill-missing-badge">Missing Skill</div>
                                <h4 className="skill-name">{area.skill}</h4>
                                <p className="skill-why"><strong>Why you need this:</strong> {area.why}</p>
                            </div>
                            <div className="resources-section">
                                <span className="resources-label">Recommended Resources:</span>
                                <div className="resource-list">
                                    {area.resources.map((resource, idx) => (
                                        <a
                                            key={idx}
                                            href={resource.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="resource-item"
                                        >
                                            <span className="resource-icon">{getResourceIcon(resource.type)}</span>
                                            <div className="resource-details">
                                                <span className="resource-title">{resource.title}</span>
                                                <span className="resource-provider">{resource.provider}</span>
                                            </div>
                                            <span className="resource-type-badge">{resource.type}</span>
                                        </a>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    // Fallback if no data
    return (
        <div className="learning-panel">
            <div className="learning-coming-soon">
                <div className="coming-soon-icon">
                    <FaGraduationCap />
                </div>
                <h3 className="coming-soon-title">No Roadmap Available</h3>
                <p className="coming-soon-description">
                    Upload and analyze a resume to see your personalized learning roadmap.
                </p>
            </div>
        </div>
    );
};

export default LearningPanel;
