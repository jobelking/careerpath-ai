/**
 * JSearch Job API Service (RapidAPI)
 * Documentation: https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
 * 
 * This service provides job listings for the Philippines using the JSearch API.
 */

const RAPIDAPI_KEY = import.meta.env.VITE_RAPIDAPI_KEY;
const JSEARCH_BASE_URL = 'https://jsearch.p.rapidapi.com';

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
    employmentType = '',
    remoteOnly = false
}) => {
    try {
        if (!RAPIDAPI_KEY) {
            console.warn('RapidAPI key not configured');
            return { results: [], count: 0, error: 'API not configured' };
        }

        // Build the search query with location
        const searchQuery = location ? `${query} in ${location}` : query;

        // Build query parameters
        const params = new URLSearchParams({
            query: searchQuery,
            page: page.toString(),
            num_pages: '1',
            country: 'ph',
            date_posted: datePosted
        });

        // Add optional filters
        if (employmentType) {
            params.append('employment_types', employmentType);
        }
        if (remoteOnly) {
            params.append('remote_jobs_only', 'true');
        }

        const url = `${JSEARCH_BASE_URL}/search?${params.toString()}`;

        console.log('🔍 Fetching jobs from JSearch:', url);

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': 'jsearch.p.rapidapi.com'
            }
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ JSearch API error:', response.status, response.statusText, errorText);
            throw new Error(`JSearch API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log('✅ JSearch API success:', data.data?.length || 0, 'jobs found');

        // Transform JSearch data to our format
        const transformedResults = (data.data || []).slice(0, resultsPerPage).map((job) => ({
            id: job.job_id,
            title: job.job_title,
            company: job.employer_name,
            companyLogo: job.employer_logo,
            location: formatLocation(job),
            type: formatEmploymentType(job.job_employment_type),
            salary: formatSalary(job.job_min_salary, job.job_max_salary, job.job_salary_currency, job.job_salary_period),
            posted: formatDate(job.job_posted_at_datetime_utc),
            description: job.job_description,
            highlights: job.job_highlights || {},
            url: job.job_apply_link,
            isRemote: job.job_is_remote,
            tags: extractTags(job),
            // Additional useful fields
            experience: job.job_required_experience || {},
            education: job.job_required_education || {},
            benefits: job.job_benefits || []
        }));

        return {
            results: transformedResults,
            count: data.data?.length || 0,
            page: page,
            status: data.status
        };

    } catch (error) {
        console.error('Error fetching jobs from JSearch:', error);
        return {
            results: [],
            count: 0,
            error: error.message
        };
    }
};

/**
 * Get job details by job ID
 * 
 * @param {string} jobId - The job ID from search results
 * @returns {Promise<Object>} Job details
 */
export const getJobDetails = async (jobId) => {
    try {
        if (!RAPIDAPI_KEY) {
            return { error: 'API not configured' };
        }

        const url = `${JSEARCH_BASE_URL}/job-details?job_id=${encodeURIComponent(jobId)}`;

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': 'jsearch.p.rapidapi.com'
            }
        });

        if (!response.ok) {
            throw new Error(`JSearch API error: ${response.status}`);
        }

        const data = await response.json();
        return data.data?.[0] || null;

    } catch (error) {
        console.error('Error fetching job details:', error);
        return { error: error.message };
    }
};

/**
 * Get estimated salary for a job title in the Philippines
 * 
 * @param {string} jobTitle - Job title to get salary estimate for
 * @param {string} location - Location (default: "Philippines")
 * @returns {Promise<Object>} Salary estimate data
 */
export const getEstimatedSalary = async (jobTitle, location = 'Philippines') => {
    try {
        if (!RAPIDAPI_KEY) {
            return { error: 'API not configured' };
        }

        const params = new URLSearchParams({
            job_title: jobTitle,
            location: location,
            radius: '100'
        });

        const url = `${JSEARCH_BASE_URL}/estimated-salary?${params.toString()}`;

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': 'jsearch.p.rapidapi.com'
            }
        });

        if (!response.ok) {
            throw new Error(`JSearch API error: ${response.status}`);
        }

        const data = await response.json();
        return data.data || [];

    } catch (error) {
        console.error('Error fetching salary estimate:', error);
        return { error: error.message };
    }
};

/**
 * Format location from job data
 */
const formatLocation = (job) => {
    if (job.job_is_remote) {
        return job.job_city ? `${job.job_city}, ${job.job_country} (Remote)` : 'Remote';
    }

    const parts = [];
    if (job.job_city) parts.push(job.job_city);
    if (job.job_state) parts.push(job.job_state);
    if (job.job_country) parts.push(job.job_country);

    return parts.join(', ') || 'Location not specified';
};

/**
 * Format employment type
 */
const formatEmploymentType = (type) => {
    const typeMap = {
        'FULLTIME': 'Full-time',
        'PARTTIME': 'Part-time',
        'CONTRACTOR': 'Contract',
        'INTERN': 'Internship',
        'TEMPORARY': 'Temporary'
    };
    return typeMap[type] || type || 'Full-time';
};

/**
 * Format salary range
 */
const formatSalary = (min, max, currency = 'PHP', period = 'YEAR') => {
    if (!min && !max) return 'Salary not specified';

    const currencySymbol = currency === 'PHP' ? '₱' : currency === 'USD' ? '$' : currency;
    const periodLabel = period === 'YEAR' ? '/year' : period === 'MONTH' ? '/month' : period === 'HOUR' ? '/hour' : '';

    const formatAmount = (amount) => {
        if (amount >= 1000000) {
            return `${(amount / 1000000).toFixed(1)}M`;
        }
        if (amount >= 1000) {
            return `${Math.round(amount / 1000)}K`;
        }
        return amount.toLocaleString();
    };

    if (min && max) {
        return `${currencySymbol}${formatAmount(min)} - ${currencySymbol}${formatAmount(max)}${periodLabel}`;
    } else if (min) {
        return `From ${currencySymbol}${formatAmount(min)}${periodLabel}`;
    } else {
        return `Up to ${currencySymbol}${formatAmount(max)}${periodLabel}`;
    }
};

/**
 * Format date to relative time
 */
const formatDate = (dateString) => {
    if (!dateString) return 'Recently posted';

    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return '1 day ago';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return date.toLocaleDateString();
};

/**
 * Extract tags from job data
 */
const extractTags = (job) => {
    const tags = [];

    if (job.job_is_remote) tags.push('Remote');
    if (job.job_employment_type) tags.push(formatEmploymentType(job.job_employment_type));
    if (job.employer_name) tags.push(job.employer_name);

    // Add some skills if available
    if (job.job_required_skills) {
        tags.push(...job.job_required_skills.slice(0, 3));
    }

    return tags.slice(0, 5); // Limit to 5 tags
};
