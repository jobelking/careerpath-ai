import React, { useState, useEffect, useCallback } from 'react';
import { FaMapMarkerAlt, FaBuilding, FaClock, FaExternalLinkAlt, FaSpinner, FaRedo, FaBriefcase } from 'react-icons/fa';
import './JobsPanel.css';
import { searchJobs } from '../../../../services/jsearchService';

const USER_FRIENDLY_ERROR = 'Job search is temporarily unavailable. Please try again later.';

/**
 * Formats a posted date string into a relative time like "2 weeks ago"
 */
const formatPostedDate = (posted) => {
    if (!posted || posted === 'Recently') return 'Recently';
    const date = new Date(posted);
    if (isNaN(date.getTime())) return posted;
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    if (diffDays < 1) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    const diffWeeks = Math.floor(diffDays / 7);
    if (diffWeeks === 1) return '1 week ago';
    if (diffWeeks < 5) return `${diffWeeks} weeks ago`;
    const diffMonths = Math.floor(diffDays / 30);
    if (diffMonths === 1) return '1 month ago';
    return `${diffMonths} months ago`;
};

// Module-level cache to persist data across component mounts/unmounts
const jobsCache = {
    careerPath: null,
    jobs: [],
    page: 1,
    hasMore: true,
    hasFetched: false,
    isFetching: false,  // Add flag to prevent concurrent fetches
    searchQuery: null
};

/**
 * JobsPanel - Displays job opportunities relevant to the career path
 * Fetches real job listings from JSearch API for the Philippines
 * Uses module-level cache to avoid re-fetching when drawer is reopened
 * 
 * @param {Object} props
 * @param {string} props.careerPath - The matched career path name
 * @param {string[]} props.jobRoles - Array of job roles for this career path
 */
const JobsPanel = ({ careerPath, jobRoles = [] }) => {
    // Initialize state from cache if available for this career path
    const [jobs, setJobs] = useState(() => {
        if (jobsCache.careerPath === careerPath && jobsCache.hasFetched) {
            return jobsCache.jobs;
        }
        return [];
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [page, setPage] = useState(() => {
        if (jobsCache.careerPath === careerPath && jobsCache.hasFetched) {
            return jobsCache.page;
        }
        return 1;
    });
    const [hasMore, setHasMore] = useState(() => {
        if (jobsCache.careerPath === careerPath && jobsCache.hasFetched) {
            return jobsCache.hasMore;
        }
        return true;
    });
    const [activeSearchQuery, setActiveSearchQuery] = useState(() => {
        if (jobsCache.careerPath === careerPath && jobsCache.hasFetched) {
            return jobsCache.searchQuery;
        }
        return null;
    });

    // Build prioritized search queries for resilience:
    // 1) all roles, 2) top 3 roles, 3) career path fallback
    const buildSearchQueries = useCallback(() => {
        const queries = [];

        if (jobRoles && jobRoles.length > 0) {
            const normalizedRoles = jobRoles
                .map((role) => role?.trim())
                .filter(Boolean);

            if (normalizedRoles.length > 0) {
                queries.push(normalizedRoles.join(' OR '));
            }

            if (normalizedRoles.length > 3) {
                queries.push(normalizedRoles.slice(0, 3).join(' OR '));
            }
        }

        const careerFallback = careerPath.replace(' Careers', '').trim();
        if (careerFallback) {
            queries.push(careerFallback);
        }

        return [...new Set(queries)];
    }, [jobRoles, careerPath]);

    // Fetch jobs from JSearch API - defined without state dependencies to avoid re-creation
    const doFetchJobs = async (pageNum, append, currentJobs) => {
        // Prevent concurrent fetches
        if (jobsCache.isFetching && !append) {
            console.log('⚠️ Fetch already in progress, skipping...');
            return;
        }

        try {
            jobsCache.isFetching = true;
            setLoading(true);
            setError(null);

            let result = null;
            let selectedQuery = activeSearchQuery;

            if (append && activeSearchQuery) {
                // Keep pagination consistent with the query that produced page 1.
                result = await searchJobs({
                    query: activeSearchQuery,
                    location: 'Philippines',
                    page: pageNum,
                    resultsPerPage: 10
                });
            } else {
                const searchQueries = buildSearchQueries();

                for (const candidateQuery of searchQueries) {
                    console.log(`🔍 Searching for "${candidateQuery}" jobs in Philippines...`);

                    const candidateResult = await searchJobs({
                        query: candidateQuery,
                        location: 'Philippines',
                        page: pageNum,
                        resultsPerPage: 10
                    });

                    result = candidateResult;
                    selectedQuery = candidateQuery;

                    if (candidateResult.error) {
                        continue;
                    }

                    if ((candidateResult.results || []).length > 0) {
                        break;
                    }
                }
            }

            if (result.error) {
                throw new Error(result.error);
            }

            let newJobs;
            if (append) {
                newJobs = [...currentJobs, ...result.results];
            } else {
                newJobs = result.results;
            }

            const newHasMore = result.results.length >= 10;

            // Update state
            setJobs(newJobs);
            setHasMore(newHasMore);
            setPage(pageNum);
            setActiveSearchQuery(selectedQuery);

            // Update cache
            jobsCache.careerPath = careerPath;
            jobsCache.jobs = newJobs;
            jobsCache.page = pageNum;
            jobsCache.hasMore = newHasMore;
            jobsCache.hasFetched = true;
            jobsCache.searchQuery = selectedQuery;

        } catch (err) {
            console.error('Failed to fetch jobs:', err);
            setError(USER_FRIENDLY_ERROR);
        } finally {
            setLoading(false);
            jobsCache.isFetching = false;
        }
    };

    // Effect to fetch jobs on mount or career path change
    useEffect(() => {
        // Skip if already cached for this career path
        if (jobsCache.careerPath === careerPath && jobsCache.hasFetched) {
            return;
        }

        // Skip if already fetching
        if (jobsCache.isFetching) {
            return;
        }

        // Reset cache if career path changed
        if (jobsCache.careerPath !== careerPath) {
            jobsCache.careerPath = careerPath;
            jobsCache.jobs = [];
            jobsCache.page = 1;
            jobsCache.hasMore = true;
            jobsCache.hasFetched = false;
            jobsCache.searchQuery = null;
            setActiveSearchQuery(null);
        }

        // Fetch if needed
        if (careerPath && !jobsCache.hasFetched) {
            doFetchJobs(1, false, []);
        }
    }, [careerPath, buildSearchQueries]);

    const handleApply = (job) => {
        // Open job URL in new tab
        if (job.url) {
            window.open(job.url, '_blank', 'noopener,noreferrer');
        }
    };

    const handleRetry = () => {
        // Clear cache for retry
        jobsCache.hasFetched = false;
        jobsCache.isFetching = false;
        jobsCache.searchQuery = null;
        setActiveSearchQuery(null);
        doFetchJobs(1, false, []);
    };

    const handleLoadMore = () => {
        doFetchJobs(page + 1, true, jobs);
    };

    // Loading state (only show on initial load, not when using cache)
    if (loading && jobs.length === 0) {
        return (
            <div className="jobs-panel">
                <div className="jobs-panel-loading">
                    <FaSpinner className="jobs-spinner" />
                    <p>Searching for jobs in the Philippines...</p>
                </div>
            </div>
        );
    }

    // Error state
    if (error && jobs.length === 0) {
        return (
            <div className="jobs-panel">
                <div className="jobs-panel-error">
                    <FaBriefcase className="jobs-error-icon" />
                    <p>Job search is temporarily unavailable. Please try again later.</p>
                    <button className="jobs-retry-btn" onClick={handleRetry}>
                        <FaRedo /> Try Again
                    </button>
                </div>
            </div>
        );
    }

    // No jobs found (after fetch completed)
    if (!loading && jobs.length === 0 && jobsCache.hasFetched) {
        return (
            <div className="jobs-panel">
                <div className="jobs-panel-empty">
                    <FaBriefcase className="jobs-empty-icon" />
                    <p>No job listings found for this career path in the Philippines.</p>
                    <button className="jobs-retry-btn" onClick={handleRetry}>
                        <FaRedo /> Search Again
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="jobs-panel">
            <div className="jobs-panel-intro">
                <p className="jobs-panel-description">
                    Job opportunities in the <strong>Philippines</strong> for <strong>{careerPath || 'your career path'}</strong>
                </p>
                <span className="jobs-count">
                    {jobs.length} position{jobs.length !== 1 ? 's' : ''} found
                </span>
            </div>

            <div className="jobs-list">
                {jobs.map((job) => (
                    <div
                        key={job.id}
                        className={`job-card ${job.isRemote ? 'job-card--remote' : ''}`}
                    >
                        {job.isRemote && (
                            <span className="job-remote-badge">Remote</span>
                        )}

                        <div className="job-card-header">
                            {job.companyLogo && (
                                <img
                                    src={job.companyLogo}
                                    alt={job.company}
                                    className="job-company-logo"
                                    onError={(e) => e.target.style.display = 'none'}
                                />
                            )}
                            <div className="job-title-section">
                                <h3 className="job-title">{job.title}</h3>
                                {job.salary && job.salary !== 'Salary not specified' && (
                                    <span className="job-salary">{job.salary}</span>
                                )}
                            </div>
                        </div>

                        <div className="job-company">
                            <FaBuilding className="job-icon" />
                            <span>{job.company}</span>
                        </div>

                        <div className="job-meta">
                            <div className="job-meta-item">
                                <FaMapMarkerAlt className="job-icon" />
                                <span>{job.location}</span>
                            </div>
                            <div className="job-meta-item">
                                <FaClock className="job-icon" />
                                <span>{formatPostedDate(job.posted)}</span>
                            </div>
                        </div>

                        <div className="job-tags">
                            <span className="job-tag job-tag--type">{job.type}</span>
                            {job.tags && job.tags.slice(0, 3).map((tag, index) => (
                                <span key={index} className="job-tag">{tag}</span>
                            ))}
                        </div>

                        {job.url && (
                            <button
                                className="job-apply-btn"
                                onClick={() => handleApply(job)}
                            >
                                Apply Now
                                <FaExternalLinkAlt className="job-apply-icon" />
                            </button>
                        )}
                    </div>
                ))}
            </div>

            {/* Load More Button */}
            {hasMore && (
                <div className="jobs-panel-footer">
                    <button
                        className="jobs-load-more-btn"
                        onClick={handleLoadMore}
                        disabled={loading}
                    >
                        {loading ? (
                            <>
                                <FaSpinner className="jobs-spinner-btn" />
                                Loading...
                            </>
                        ) : (
                            'Load More Jobs'
                        )}
                    </button>
                </div>
            )}
        </div>
    );
};

export default JobsPanel;
