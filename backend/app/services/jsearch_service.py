"""
JSearch API Service (RapidAPI)
Handles job search requests via RapidAPI's JSearch endpoint
"""

import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

JSEARCH_BASE_URL = 'https://jsearch.p.rapidapi.com'
RAPIDAPI_HOST = 'jsearch.p.rapidapi.com'


def get_rapidapi_keys() -> List[str]:
    """Get all configured RapidAPI keys for fallback."""
    keys = []
    
    for i in range(1, 5):
        env_var = "RAPIDAPI_KEY" if i == 1 else f"RAPIDAPI_KEY_{i}"
        key = os.getenv(env_var)
        if key:
            keys.append(key)
    
    return keys


def search_jobs(
    query: str,
    location: str = "Philippines",
    page: int = 1,
    results_per_page: int = 10,
    date_posted: Optional[str] = None,
    employment_type: Optional[str] = None,
    remote_only: bool = False
) -> Dict[str, Any]:
    """
    Search for jobs using JSearch API.
    
    Args:
        query: Job search query (e.g., "Software Developer")
        location: Location (default: "Philippines")
        page: Page number (default: 1)
        results_per_page: Results per page (default: 10)
        date_posted: Filter by date: 'all', 'today', '3days', 'week', 'month'
        employment_type: Filter: 'FULLTIME', 'PARTTIME', 'CONTRACTOR', 'INTERN'
        remote_only: Filter for remote jobs only
        
    Returns:
        Dict with job search results or error
        
    Raises:
        Exception: If API call fails with all keys
    """
    rapidapi_keys = get_rapidapi_keys()
    
    if not rapidapi_keys:
        raise Exception("No RapidAPI keys configured")
    
    # Build query parameters
    params = {
        'query': query,
        'page': str(page),
        'num_pages': '1',
        'date_posted': date_posted or 'all'
    }
    
    if location:
        params['query'] = f"{query} in {location}"
    
    if employment_type:
        params['employment_types'] = employment_type
    
    if remote_only:
        params['remote_jobs_only'] = 'true'
    
    # Try each key until one works
    last_error = None
    
    for i, key in enumerate(rapidapi_keys):
        try:
            headers = {
                'x-rapidapi-key': key,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            
            response = requests.get(
                f"{JSEARCH_BASE_URL}/search",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                if i > 0:
                    print(f"⚠️ JSearch succeeded using fallback API key #{i + 1}")
                
                data = response.json()
                
                # Transform results to match frontend expectations
                results = []
                for job in data.get('data', []):
                    # Format salary information from available fields
                    job_salary = job.get('job_salary')
                    min_salary = job.get('job_min_salary')
                    max_salary = job.get('job_max_salary')
                    currency = job.get('job_salary_currency') or ''
                    period = job.get('job_salary_period') or ''
                    if period:
                        period = f"/{period.lower()}"
                    
                    salary_str = 'Salary not specified'
                    if job_salary is not None:
                        salary_str = f"{job_salary} {currency}{period}".strip()
                    elif min_salary is not None and max_salary is not None:
                        salary_str = f"{min_salary} - {max_salary} {currency}{period}".strip()
                    elif min_salary is not None:
                        salary_str = f"{min_salary}+ {currency}{period}".strip()
                    elif max_salary is not None:
                        salary_str = f"Up to {max_salary} {currency}{period}".strip()

                    results.append({
                        'id': job.get('job_id'),
                        'title': job.get('job_title'),
                        'company': job.get('employer_name'),
                        'location': job.get('job_city') or job.get('job_country', 'Not specified'),
                        'salary': salary_str,
                        'type': job.get('job_employment_type', 'Full-time'),
                        'posted': job.get('job_posted_at_datetime_utc', 'Recently'),
                        'url': job.get('job_apply_link'),
                        'companyLogo': job.get('employer_logo'),
                        'isRemote': job.get('job_is_remote', False),
                        'tags': job.get('job_highlights', {}).get('Qualifications', [])[:3]
                    })
                
                return {
                    'results': results,
                    'error': None
                }
            
            # Handle errors
            last_error = Exception(f"JSearch API error: {response.status_code} {response.text}")
            
            # Only retry with next key for auth/rate limit errors
            if response.status_code not in [401, 403, 429]:
                raise last_error
            
            if i < len(rapidapi_keys) - 1:
                print(f"⚠️ JSearch key #{i + 1} failed ({response.status_code}). Trying next key...")
                
        except requests.exceptions.Timeout:
            last_error = Exception("JSearch API request timed out")
            if i < len(rapidapi_keys) - 1:
                print(f"⚠️ JSearch request timed out on key #{i + 1}. Trying next key...")
        except requests.exceptions.RequestException as e:
            last_error = Exception(f"Network error calling JSearch API: {str(e)}")
            if i < len(rapidapi_keys) - 1:
                print(f"⚠️ JSearch request failed on key #{i + 1}. Trying next key...")
    
    # All keys failed
    raise last_error or Exception("JSearch request failed with all keys")
