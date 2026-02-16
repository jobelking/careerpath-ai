/**
 * JSearch Job API Service - Backend Proxy
 * Now calls the backend /api/jobs endpoint instead of RapidAPI directly
 */

import apiService from './api/apiService';

const USER_FRIENDLY_ERROR = 'Job search is temporarily unavailable. Please try again later.';

/**
 * Search for jobs based on career path and location
 * 
 * @param {Object} params
 * @param {string} params.query - Job search query (e.g., "Software Developer")
 * @param {string} params.location - Location (default: "Philippines")
 * @param {number} params.page - Page number (default: 1)
 * @param {number} params.resultsPerPage - Results per page (default: 10)
 * @param {string} params.datePosted - Filter by date: 'all', 'today', '3days', 'week', 'month'
 * @param {string} params.employmentType - Filter: 'FULLTIME', 'PARTTIME', 'CONTRACTOR', 'INTERN'
 * @param {boolean} params.remoteOnly - Filter for remote jobs only
 * @returns {Promise<Object>} Job search results
 */
export const searchJobs = async ({
    query,
    location = 'Philippines',
    page = 1,
    resultsPerPage = 10,
    datePosted = 'all',
    employmentType,
    remoteOnly = false
}) => {
    try {
        // Call backend endpoint
        const response = await apiService.searchJobs({
            query,
            location,
            page,
            resultsPerPage,
            datePosted,
            employmentType,
            remoteOnly
        });

        return response;
    } catch (error) {
        console.error('Job search error:', error);
        return {
            results: [],
            error: USER_FRIENDLY_ERROR
        };
    }
};
