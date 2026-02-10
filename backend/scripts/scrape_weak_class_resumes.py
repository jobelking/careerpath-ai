"""
Scrape Resumes for Weak Career Path Classes
=============================================
Scrapes real resumes from postjobfree.com for job roles related to
weak career path classes. Uses the SAME process as ResumeScraper.py.

- Searches by job role title (no location filter)
- Saves RAW resume text (no preprocessing)
- path column = the job title from the scraped resume
- Separate CSV file per career path class

Usage:
    python execution/scrape_weak_class_resumes.py
    python execution/scrape_weak_class_resumes.py --classes healthcare hr teacher
    python execution/scrape_weak_class_resumes.py --max-per-class 50
    python execution/scrape_weak_class_resumes.py --pages 5

Output:
    .tmp/healthcare.csv
    .tmp/hr.csv
    .tmp/teacher.csv
    ... etc
"""

import os
import sys
import csv
import time
import hashlib
import argparse
import logging
from typing import List, Dict, Set

# ---------------------------------------------------------------------------
# Imports (same as ResumeScraper.py)
# ---------------------------------------------------------------------------
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: Missing required packages. Install them with:")
    print("  pip install requests beautifulsoup4")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, ".tmp")

BASE_URL = "https://www.postjobfree.com"
MIN_RESUME_LENGTH = 200       # chars — skip empty/broken pages
DEFAULT_MAX_PER_CLASS = 100   # resumes per class (collects up to max available)
DEFAULT_PAGES = 5             # search result pages to crawl per job role
REQUEST_DELAY = 3             # seconds between requests (same as ResumeScraper.py)

# ---------------------------------------------------------------------------
# Weak classes → job-role search terms
# These are JOB ROLES related to each career path.
# The scraper will search for EACH role separately.
# ---------------------------------------------------------------------------
WEAK_CLASS_JOB_ROLES: Dict[str, List[str]] = {
"healthcare": [
        "registered nurse",
        "nurse practitioner",
        "licensed practical nurse",
        "medical assistant",
        "clinical coordinator",
        "pharmacy technician",
        "healthcare administrator",
        "patient care coordinator",
        "physical therapist",
        "occupational therapist",
        "healthcare manager",
        "medical records clerk",
        "certified nursing assistant",
        "physician assistant",
        "respiratory therapist",
        "radiology technician",
        "surgical technologist",
        "emergency medical technician",
        "paramedic",
        "medical coder",
        "health information technician",
        "clinical research coordinator",
        "home health aide",
        "dental hygienist",
        "medical laboratory technician",
    ],
"engineering": [
        "mechanical engineer",
        "civil engineer",
        "electrical engineer",
        "chemical engineer",
        "industrial engineer",
        "manufacturing engineer",
        "structural engineer",
        "project engineer",
        "design engineer",
        "process engineer",
        "quality engineer",
        "production engineer",
        "maintenance engineer",
        "reliability engineer",
        "systems engineer",
        "test engineer",
        "field engineer",
        "plant engineer",
        "facilities engineer",
        "automation engineer",
        "instrumentation engineer",
        "environmental engineer",
        "petroleum engineer",
        "mining engineer",
    ],
    "business-development": [
        "business development manager",
        "business development executive",
        "business development representative",
        "partnership manager",
        "client acquisition specialist",
        "strategic alliance manager",
        "growth manager",
        "business development associate",
    ],
    "public-relations": [
        "public relations specialist",
        "communications officer",
        "media relations coordinator",
        "publicity manager",
        "corporate communications",
        "public relations manager",
        "media coordinator",
        "PR manager",
    ],
    "aviation": [
        "aviation",
        "flight engineer",
        "aircraft maintenance technician",
        "air traffic controller",
        "cabin crew",
        "flight attendant",
        "aerospace technician",
        "aviation mechanic",
        "airline operations",
    ],
    "fitness": [
        "fitness instructor",
        "gym trainer",
        "yoga instructor",
        "strength conditioning coach",
        "wellness coordinator",
        "fitness coach",
        "group fitness instructor",
        "pilates instructor",
        "zumba instructor",
        "crossfit trainer",
        "aerobics instructor",
        "spinning instructor",
        "athletic trainer",
        "exercise physiologist",
        "health coach",
        "recreation coordinator",
    ],
    "hr": [
        "human resources manager",
        "HR executive",
        "recruitment specialist",
        "talent acquisition specialist",
        "HR generalist",
        "HR coordinator",
        "compensation benefits analyst",
        "people operations manager",
        "recruiter",
        "HR business partner",
        "employee relations specialist",
        "talent management specialist",
        "HR analyst",
        "benefits administrator",
        "payroll specialist",
        "learning development specialist",
        "training coordinator",
        "onboarding specialist",
        "HR assistant",
        "workforce planning analyst",
    ],
    "teacher": [
        "school teacher",
        "high school teacher",
        "elementary teacher",
        "math teacher",
        "english teacher",
        "tutor",
        "lecturer",
        "education coordinator",
        "curriculum developer",
        "teaching assistant",
    ],
    "design-creative": [
        "graphic designer",
        "UI UX designer",
        "visual designer",
        "creative director",
        "art director",
        "web designer",
        "motion graphics designer",
        "illustrator",
        "product designer",
    ],
    "consultant": [
        "management consultant",
        "strategy consultant",
        "business consultant",
        "technology consultant",
        "SAP consultant",
        "consulting analyst",
        "engagement manager",
    ],
    "advocate": [
        "lawyer",
        "attorney",
        "legal advisor",
        "legal counsel",
        "litigation associate",
        "corporate lawyer",
        "paralegal",
        "legal assistant",
        "law clerk",
    ],
    "chef": [
        "chef",
        "sous chef",
        "pastry chef",
        "executive chef",
        "line cook",
        "head cook",
        "kitchen manager",
        "catering manager",
        "food service manager",
        "culinary specialist",
    ],
    "business-development": [
        "business development manager",
        "business development executive",
        "business development representative",
        "partnership manager",
        "client acquisition specialist",
        "strategic alliance manager",
        "growth manager",
        "business development associate",
        "business development director",
        "channel partner manager",
        "strategic partnerships director",
        "enterprise sales",
        "account development manager",
        "market development manager",
        "relationship manager",
    ],
}


# ===========================================================================
# Helpers
# ===========================================================================

def text_hash(text: str) -> str:
    """MD5 hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# Job title exclusion filters - keywords that indicate wrong career path
JOB_TITLE_EXCLUSIONS = {
    "aviation": ["driver", "truck", "delivery", "school teacher", "high school", "university", "student"],
    "fitness": ["high school", "school teacher", "university", "student", "academic"],
    "healthcare": ["veterinary", "animal", "pet", "vet tech", "veterinarian"],
    "teacher": ["trainer", "fitness", "gym"],
    "consultant": ["insurance sales", "sales representative"],
    "hr": ["recruitment agency owner", "staffing agency owner", "headhunter"],
    "engineering": ["software engineer", "data engineer", "devops", "cloud engineer", "network engineer"],
    "business-development": ["software", "developer", "programmer", "data scientist"],
}

def is_valid_job_title(job_title: str, career_path: str) -> bool:
    """
    Validate if job title is relevant to the career path.
    Filters out mixed/irrelevant results.
    """
    title_lower = job_title.lower()
    
    # Check career-specific exclusions
    if career_path in JOB_TITLE_EXCLUSIONS:
        for exclusion in JOB_TITLE_EXCLUSIONS[career_path]:
            if exclusion in title_lower:
                return False
    
    return True


# ===========================================================================
# PostJobFree scraper — SAME PROCESS as ResumeScraper.py
# ===========================================================================

def get_resume_links(search_term: str, page: int = 1) -> List[str]:
    """
    Search postjobfree.com for resumes matching a job role.
    NO location filter — searches globally.
    Returns a list of full resume URLs from one page of results.
    """
    # Same URL pattern as ResumeScraper.py but:
    #   - t= is the job role title
    #   - l= is EMPTY (no location filter, no radius)
    #   - r=100 results per page
    url = (
        f"{BASE_URL}/resumes?"
        f"q=&n=&t={requests.utils.quote(search_term)}&d=&l=&r=100"
        f"&p={page}"
    )
    try:
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            log.warning(f"  Search page returned {response.status_code}: {url}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        title_tags = soup.find_all('h3', attrs={'class': 'itemTitle'})

        # Same link extraction as ResumeScraper.py
        links = [BASE_URL + title_tag.a['href'] for title_tag in title_tags if title_tag.a]
        return links

    except Exception as e:
        log.warning(f"  Error fetching search page: {e}")
        return []


def extract_resume(url: str, career_path: str) -> Dict[str, str] | None:
    """
    Fetch a single resume page — SAME PROCESS as ResumeScraper.py:
      - job_title from div.innercontent > h1 (used for validation only)
      - resume text from div.normalText, raw, [:-23] to trim footer

    Returns dict with 'path' (career_path class), 'text_needed' (raw resume text),
    'job_title' (for validation), and 'url', or None if extraction fails.
    """
    try:
        res = requests.get(url, timeout=20)
        if res.status_code != 200:
            log.debug(f"    HTTP {res.status_code}: {url}")
            return None

        content = BeautifulSoup(res.content, 'html.parser')

        # Extract job title — used for validation and logging
        innercontent = content.find('div', attrs={'class': 'innercontent'})
        if not innercontent or not innercontent.find('h1'):
            log.debug(f"    No job title found: {url}")
            return None
        job_title = innercontent.find('h1').get_text()

        # Extract resume text — same as ResumeScraper.py
        # RAW .get_text()[:-23] — no preprocessing, no cleaning
        normal_text = content.find('div', attrs={'class': 'normalText'})
        if not normal_text:
            log.debug(f"    No resume text found: {url}")
            return None

        resume_text = normal_text.get_text()[:-23]  # trim footer, same as ResumeScraper.py

        if len(resume_text) < MIN_RESUME_LENGTH:
            return None

        return {
            'path': career_path,         # path = career class (e.g., "aviation", "fitness")
            'text_needed': resume_text,  # raw text, no preprocessing
            'job_title': job_title,      # original job title for validation
            'url': url                   # source URL for reference
        }

    except Exception as e:
        log.debug(f"    Error extracting resume: {e}")
        return None


# ===========================================================================
# Per-class scraping orchestrator
# ===========================================================================

def scrape_class(
    career_path: str,
    job_roles: List[str],
    max_resumes: int,
    pages_per_role: int,
    seen_hashes: Set[str],
) -> List[Dict[str, str]]:
    """
    Scrape resumes for one career path class across all its job roles.
    """
    results = []
    log.info(f"▶ Scraping class: {career_path} ({len(job_roles)} job roles, target {max_resumes})")

    for role in job_roles:
        if len(results) >= max_resumes:
            break

        log.info(f"  🔍 Role: '{role}'")
        
        seen_links = set()  # Track links for this role to detect duplicate pages

        for page in range(1, pages_per_role + 1):
            if len(results) >= max_resumes:
                break

            links = get_resume_links(role, page=page)
            if not links:
                log.info(f"    Page {page}: no results, moving on")
                break

            # Check if this page has same links as previous (duplicate page)
            links_set = set(links)
            if links_set.issubset(seen_links):
                log.info(f"    Page {page}: duplicate of previous page, moving to next role")
                break
            
            seen_links.update(links_set)
            log.info(f"    Page {page}: found {len(links)} resume links")

            for link in links:
                if len(results) >= max_resumes:
                    break

                # Same process as ResumeScraper.py
                result = extract_resume(link, career_path)
                if result:
                    # Validate job title is relevant to career path
                    if not is_valid_job_title(result['job_title'], career_path):
                        log.debug(f"    ✗ filtered out: {result['job_title']}")
                        continue
                    
                    h = text_hash(result['text_needed'])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        results.append(result)
                        log.info(
                            f"    ✓ #{len(results):3d} | {result['job_title'][:50]:50s} | "
                            f"{len(result['text_needed']):,} chars | {result['url']}"
                        )
                    else:
                        log.debug(f"    ✗ duplicate: {link}")

                # Same delay as ResumeScraper.py: time.sleep(3)
                time.sleep(REQUEST_DELAY)

    log.info(f"  ✅ Collected {len(results)} resumes for '{career_path}'\n")
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scrape real resumes from postjobfree.com for weak career path classes"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Specific classes to scrape (default: all). E.g. --classes healthcare hr teacher",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS,
        help=f"Max resumes per class (default: {DEFAULT_MAX_PER_CLASS})",
    )
    parser.add_argument(
        "--pages", type=int, default=DEFAULT_PAGES,
        help=f"Search result pages to crawl per job role (default: {DEFAULT_PAGES})",
    )
    args = parser.parse_args()

    # Determine which classes to scrape
    if args.classes:
        classes_to_scrape = {
            k: v for k, v in WEAK_CLASS_JOB_ROLES.items() if k in args.classes
        }
        invalid = set(args.classes) - set(WEAK_CLASS_JOB_ROLES.keys())
        if invalid:
            log.error(f"Unknown classes: {invalid}")
            log.info(f"Available: {list(WEAK_CLASS_JOB_ROLES.keys())}")
            sys.exit(1)
    else:
        classes_to_scrape = WEAK_CLASS_JOB_ROLES

    log.info("=" * 60)
    log.info("  RESUME SCRAPER — WEAK CLASSES (postjobfree.com)")
    log.info("=" * 60)
    log.info(f"  Classes     : {list(classes_to_scrape.keys())}")
    log.info(f"  Max/class   : {args.max_per_class}")
    log.info(f"  Pages/role  : {args.pages}")
    log.info(f"  Output dir  : {OUTPUT_DIR}")
    log.info(f"  Delay       : {REQUEST_DELAY}s (same as ResumeScraper.py)")
    log.info("=" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seen_hashes: Set[str] = set()
    total_scraped = 0

    for career_path, job_roles in classes_to_scrape.items():
        results = scrape_class(
            career_path=career_path,
            job_roles=job_roles,
            max_resumes=args.max_per_class,
            pages_per_role=args.pages,
            seen_hashes=seen_hashes,
        )

        # ----- Save SEPARATE CSV per career path class -----
        if results:
            output_file = os.path.join(OUTPUT_DIR, f"{career_path}.csv")
            with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['path', 'text_needed'])
                writer.writeheader()
                for row in results:
                    # Only write path and text_needed columns
                    writer.writerow({
                        'path': row['path'],
                        'text_needed': row['text_needed']
                    })
            log.info(f"  💾 Saved {len(results)} resumes → {output_file}\n")
            total_scraped += len(results)
        else:
            log.warning(f"  ⚠ No resumes collected for '{career_path}'\n")

    log.info("=" * 60)
    log.info("  SCRAPING COMPLETE")
    log.info("=" * 60)
    log.info(f"  Total resumes : {total_scraped}")
    log.info(f"  Output dir    : {OUTPUT_DIR}")
    log.info(f"  Files created :")
    for career_path in classes_to_scrape:
        fpath = os.path.join(OUTPUT_DIR, f"{career_path}.csv")
        if os.path.exists(fpath):
            log.info(f"    ✅ {career_path}.csv")


if __name__ == "__main__":
    main()
