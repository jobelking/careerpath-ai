import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Logo from '../../components/common/Logo';
import RightDock from '../../components/common/RightDock';
import RightDrawer from '../../components/common/RightDrawer';
import { JobsPanel, LearningPanel, CertificationPanel } from '../../components/common/RightDrawer/panels';
import { useDashboard } from '../../context/DashboardContext';
import { useAuth } from '../../context/AuthContext';
import apiService from '../../services/api/apiService';
import { careerIcons } from '../../utils/careerIcons';
import { otherIcons } from '../../utils/otherIcons';
import CareerPathsModal from '../../components/common/CareerPathsModal/CareerPathsModal';
import { exportToPdf } from '../../utils/exportToPdf';
import { calculateProfileFit, normalizeTop3Fits } from '../../utils/profileFit';
import './learnmore.css';

// Career-specific content for all 26 career paths
const careerContent = {
    "Quality Assurance & Testing Careers": {
        overview: "Quality Assurance (QA) professionals ensure software products meet specified requirements and are free of defects. This career path involves designing test strategies, creating test cases, executing manual and automated tests, and collaborating with development teams to improve product quality.",
        keySkills: ["Test automation (Selenium, Cypress, Playwright)", "Manual testing and exploratory testing", "Test case design and execution", "Bug tracking and reporting (Jira, Bugzilla)", "API testing (Postman, REST Assured)", "Performance testing (JMeter, LoadRunner)", "CI/CD integration for testing"],
        jobRoles: ["QA Analyst", "Software Tester", "QA Engineer", "Test Automation Engineer", "Senior QA Lead", "Quality Assurance Manager"],
        growthAreas: ["Advanced automation frameworks", "Security testing certifications", "Performance engineering", "DevOps and shift-left testing practices"],
        marketInsight: "QA roles are essential in every software organization, with automation skills commanding premium salaries."
    },
    "Software Development Careers": {
        overview: "Software Development encompasses designing, coding, testing, and maintaining software applications. Developers work across frontend, backend, or full-stack roles, building everything from web applications to enterprise systems using various programming languages and frameworks.",
        keySkills: ["Programming languages (Python, JavaScript, Java, C++)", "Web frameworks (React, Angular, Node.js, Django)", "Database management (SQL, MongoDB)", "Version control (Git)", "API development and integration", "Software architecture and design patterns", "Agile/Scrum methodologies"],
        jobRoles: ["Junior Developer", "Software Engineer", "Full-Stack Developer", "Backend Developer", "Frontend Developer", "Senior Software Engineer", "Tech Lead", "Principal Engineer"],
        growthAreas: ["Cloud-native development", "System design and architecture", "DevOps practices", "Emerging technologies (AI/ML integration)"],
        marketInsight: "Software development remains one of the highest-demand career paths with competitive salaries across all experience levels."
    },
    "Business Analysis Careers": {
        overview: "Business Analysts bridge the gap between business needs and technical solutions. They gather requirements, analyze processes, document specifications, and work with stakeholders to ensure projects deliver expected business value.",
        keySkills: ["Requirements gathering and documentation", "Stakeholder management", "Process modeling (BPMN, UML)", "Data analysis and visualization", "Agile/Scrum methodologies", "SQL and business intelligence tools", "Communication and presentation skills"],
        jobRoles: ["Junior Business Analyst", "Business Analyst", "Senior Business Analyst", "Product Owner", "Business Systems Analyst", "Lead BA", "Director of Business Analysis"],
        growthAreas: ["Data analytics and visualization", "Product management skills", "Industry-specific domain expertise", "Agile coaching certifications"],
        marketInsight: "Business analysts are critical for digital transformation initiatives, with demand growing across industries."
    },
    "Network Administration Careers": {
        overview: "Network Administrators design, implement, and maintain organizational network infrastructure. This includes managing routers, switches, firewalls, and ensuring secure, reliable connectivity across enterprise networks.",
        keySkills: ["Network protocols (TCP/IP, DNS, DHCP)", "Router and switch configuration (Cisco, Juniper)", "Firewall management and security", "Network monitoring and troubleshooting", "VPN and remote access solutions", "Cloud networking (AWS, Azure)", "Network documentation and diagram creation"],
        jobRoles: ["Network Technician", "Network Administrator", "Network Engineer", "Senior Network Engineer", "Network Architect", "IT Infrastructure Manager"],
        growthAreas: ["Cloud networking certifications (AWS, Azure)", "Network automation (Ansible, Python)", "Software-defined networking (SDN)", "Cybersecurity specialization"],
        marketInsight: "Network professionals are essential for maintaining business operations, with cloud skills increasingly valuable."
    },
    "DevOps & Site Reliability Careers": {
        overview: "DevOps and SRE professionals bridge development and operations, automating infrastructure, managing CI/CD pipelines, and ensuring system reliability. They focus on infrastructure as code, monitoring, and continuous improvement of deployment processes.",
        keySkills: ["CI/CD pipelines (Jenkins, GitLab CI, GitHub Actions)", "Container technologies (Docker, Kubernetes)", "Infrastructure as Code (Terraform, Ansible)", "Cloud platforms (AWS, Azure, GCP)", "Monitoring and observability (Prometheus, Grafana)", "Scripting (Bash, Python)", "Linux system administration"],
        jobRoles: ["DevOps Engineer", "Site Reliability Engineer (SRE)", "Platform Engineer", "Cloud Engineer", "Senior DevOps Engineer", "DevOps Architect", "Head of Platform Engineering"],
        growthAreas: ["Kubernetes advanced certifications", "Cloud architecture", "Security automation (DevSecOps)", "Platform engineering practices"],
        marketInsight: "DevOps expertise commands premium salaries, with Kubernetes and cloud skills in highest demand."
    },
    "Data Science & AI Careers": {
        overview: "Data Scientists and AI/ML Engineers extract insights from data using statistical analysis, machine learning, and deep learning. They build predictive models, conduct experiments, and help organizations make data-driven decisions.",
        keySkills: ["Python and R programming", "Machine learning frameworks (TensorFlow, PyTorch, scikit-learn)", "Statistical analysis and modeling", "Data visualization (Matplotlib, Tableau)", "SQL and big data tools (Spark, Hadoop)", "Deep learning and neural networks", "Feature engineering and model optimization"],
        jobRoles: ["Data Analyst", "Junior Data Scientist", "Data Scientist", "Machine Learning Engineer", "Senior Data Scientist", "AI Engineer", "Principal Data Scientist", "Head of Data Science"],
        growthAreas: ["Large language models and GenAI", "MLOps and model deployment", "Deep learning specializations", "Domain-specific AI applications"],
        marketInsight: "Data science and AI roles offer some of the highest salaries in tech, with GenAI skills especially sought after."
    },
    "Cybersecurity Careers": {
        overview: "Cybersecurity professionals protect organizations from digital threats through risk assessment, security implementation, incident response, and compliance management. This field encompasses roles from security operations to ethical hacking.",
        keySkills: ["Security frameworks (NIST, ISO 27001)", "Penetration testing and vulnerability assessment", "SIEM tools (Splunk, QRadar)", "Network security and firewalls", "Incident response and forensics", "Security compliance (GDPR, HIPAA, PCI-DSS)", "Ethical hacking tools (Metasploit, Burp Suite)"],
        jobRoles: ["Security Analyst", "Cybersecurity Analyst", "Penetration Tester", "Security Engineer", "SOC Analyst", "Senior Security Engineer", "Security Architect", "CISO"],
        growthAreas: ["Cloud security certifications", "Advanced penetration testing", "Threat intelligence", "Security automation and orchestration"],
        marketInsight: "Cybersecurity has one of the largest talent gaps, with certified professionals commanding premium compensation."
    },
    "Mobile Development Careers": {
        overview: "Mobile Developers create applications for iOS and Android platforms. This includes native development, cross-platform frameworks, and ensuring optimal user experience across diverse mobile devices.",
        keySkills: ["iOS development (Swift, SwiftUI, Objective-C)", "Android development (Kotlin, Java)", "Cross-platform frameworks (React Native, Flutter)", "Mobile UI/UX design principles", "App store deployment and optimization", "Mobile app architecture (MVVM, Clean Architecture)", "RESTful API integration"],
        jobRoles: ["Junior Mobile Developer", "iOS Developer", "Android Developer", "Mobile Developer", "Senior Mobile Developer", "Mobile Architect", "Mobile Development Lead"],
        growthAreas: ["Cross-platform development mastery", "Mobile DevOps and CI/CD", "AR/VR mobile experiences", "Mobile security best practices"],
        marketInsight: "Mobile developers remain in high demand as organizations prioritize mobile-first experiences."
    },
    "Construction Careers": {
        overview: "Construction professionals manage and execute building projects from planning through completion. This includes project management, safety compliance, site supervision, and coordination of construction activities.",
        keySkills: ["Project management and scheduling", "Blueprint reading and interpretation", "Safety regulations and OSHA compliance", "Construction estimating and budgeting", "CAD software proficiency", "Building codes and permits", "Quality control and inspection"],
        jobRoles: ["Construction Worker", "Site Supervisor", "Construction Manager", "Project Manager", "Safety Manager", "Construction Estimator", "General Contractor", "Director of Construction"],
        growthAreas: ["Project management certifications (PMP)", "Sustainable building practices (LEED)", "BIM (Building Information Modeling)", "Construction technology adoption"],
        marketInsight: "Construction management roles offer stable career growth with infrastructure investment continuing to grow."
    },
    "Engineering Careers": {
        overview: "Engineering encompasses various disciplines including mechanical, electrical, civil, and chemical engineering. Engineers design, develop, and improve systems, products, and processes using scientific and mathematical principles.",
        keySkills: ["Engineering design and analysis", "CAD/CAM software (AutoCAD, SolidWorks)", "Technical documentation", "Project management", "Problem-solving and critical thinking", "Industry-specific regulations and standards", "Quality assurance and testing"],
        jobRoles: ["Junior Engineer", "Design Engineer", "Project Engineer", "Senior Engineer", "Lead Engineer", "Engineering Manager", "Principal Engineer", "VP of Engineering"],
        growthAreas: ["Advanced simulation and modeling", "Sustainable engineering practices", "Industry 4.0 technologies", "Cross-disciplinary engineering skills"],
        marketInsight: "Engineering roles offer strong job security with consistent demand across manufacturing and technology sectors."
    },
    "Design & Creative Careers": {
        overview: "Design and Creative professionals create visual solutions for communication, user interfaces, brand experiences, and artistic expression. This includes graphic design, UI/UX design, fine arts, fashion design, motion graphics, and visual communication across digital and print media.",
        keySkills: ["Graphic design (Adobe Creative Suite)", "UI/UX design (Figma, Sketch)", "Typography and color theory", "Visual communication and branding", "Prototyping and wireframing", "Motion graphics and animation", "Fine arts and illustration", "Fashion design and styling"],
        jobRoles: ["Junior Designer", "Graphic Designer", "UI Designer", "UX Designer", "Fashion Designer", "Artist", "Art Director", "Creative Director", "Head of Design"],
        growthAreas: ["Product design and strategy", "Design systems and operations", "3D design and AR/VR", "AI-assisted design tools", "Sustainable fashion practices"],
        marketInsight: "Design and creative skills are increasingly valued across industries, with UX/UI designers and creative directors in particularly high demand."
    },
    "Healthcare Careers": {
        overview: "Healthcare professionals provide patient care, clinical services, and health management across medical facilities. This field includes nursing, clinical practice, health administration, and specialized medical roles.",
        keySkills: ["Patient care and clinical skills", "Medical terminology and procedures", "Electronic health records (EHR)", "Healthcare regulations (HIPAA)", "Communication and empathy", "Critical thinking and decision-making", "Team collaboration in clinical settings"],
        jobRoles: ["Medical Assistant", "Registered Nurse", "Clinical Specialist", "Nurse Practitioner", "Healthcare Administrator", "Clinical Manager", "Director of Nursing"],
        growthAreas: ["Specialized certifications", "Healthcare technology adoption", "Telehealth and remote care", "Healthcare leadership and management"],
        marketInsight: "Healthcare offers exceptional job security with growing demand across all specializations."
    },
    "Finance & Accounting Careers": {
        overview: "Finance and Accounting professionals manage financial operations, investment analysis, banking services, and financial compliance. This includes accounting, auditing, investment management, banking operations, financial planning, and corporate finance.",
        keySkills: ["Financial accounting and bookkeeping", "Financial analysis and modeling", "Investment analysis and valuation", "Tax preparation and planning", "Banking operations and compliance", "Risk assessment and management", "GAAP/IFRS standards", "Bloomberg and financial platforms"],
        jobRoles: ["Junior Accountant", "Financial Analyst", "Auditor", "Investment Analyst", "Personal Banker", "Senior Accountant", "Portfolio Manager", "Controller", "CFO"],
        growthAreas: ["CPA/CFA certification", "FinTech and digital banking", "ESG and sustainable investing", "Data analytics in finance"],
        marketInsight: "Finance and accounting careers offer stable progression with certifications like CPA and CFA significantly boosting earning potential."
    },
    "Sales Careers": {
        overview: "Sales professionals drive revenue growth by identifying prospects, building relationships, and closing deals. This includes B2B and B2C sales, account management, and sales leadership roles.",
        keySkills: ["Prospecting and lead generation", "Negotiation and closing techniques", "CRM systems (Salesforce, HubSpot)", "Presentation and communication skills", "Relationship building and account management", "Sales forecasting and pipeline management", "Product/industry knowledge"],
        jobRoles: ["Sales Representative", "Account Executive", "Sales Manager", "Key Account Manager", "Senior Sales Executive", "Regional Sales Director", "VP of Sales"],
        growthAreas: ["Enterprise sales methodologies", "Sales technology and automation", "Strategic account management", "Sales leadership development"],
        marketInsight: "Top-performing sales professionals can earn significant commission-based income with uncapped potential."
    },
    "Fitness & Wellness Careers": {
        overview: "Fitness and Wellness professionals help clients achieve health goals through exercise programming, nutrition guidance, and wellness coaching. This includes personal training, group fitness, and wellness program management.",
        keySkills: ["Exercise physiology and programming", "Personal training and coaching", "Nutrition fundamentals", "Client assessment and goal setting", "Group fitness instruction", "Wellness program development", "CPR/First Aid certification"],
        jobRoles: ["Fitness Instructor", "Personal Trainer", "Group Fitness Instructor", "Wellness Coach", "Fitness Manager", "Gym Director", "Wellness Program Director"],
        growthAreas: ["Specialized certifications (sports, rehabilitation)", "Nutrition coaching credentials", "Online coaching and digital fitness", "Corporate wellness programs"],
        marketInsight: "Fitness careers offer flexibility and fulfillment, with growing opportunities in corporate and digital wellness."
    },
    "Education & Teaching Careers": {
        overview: "Education professionals facilitate learning through curriculum development, instruction, and student support. This includes classroom teaching, educational technology, curriculum design, and educational administration.",
        keySkills: ["Curriculum development and lesson planning", "Classroom management", "Student assessment and evaluation", "Educational technology integration", "Communication and presentation", "Differentiated instruction", "Collaboration with parents and staff"],
        jobRoles: ["Teaching Assistant", "Teacher", "Subject Specialist", "Curriculum Developer", "Instructional Coach", "Department Head", "Principal", "Director of Education"],
        growthAreas: ["Educational technology specialization", "Special education credentials", "Educational leadership degrees", "Online and hybrid teaching methods"],
        marketInsight: "Education offers rewarding careers with strong job stability, especially in specialized subject areas."
    },

    "Digital Media & Marketing Careers": {
        overview: "Digital Media and Marketing professionals create and execute strategies to reach audiences through digital channels. This includes content marketing, social media, SEO, paid advertising, and analytics.",
        keySkills: ["Digital marketing strategy", "Social media management and marketing", "Search engine optimization (SEO/SEM)", "Content creation and copywriting", "Marketing analytics (Google Analytics)", "Email marketing and automation", "Paid advertising (Google Ads, Meta Ads)"],
        jobRoles: ["Social Media Coordinator", "Digital Marketing Specialist", "Content Marketer", "SEO Specialist", "Marketing Manager", "Digital Marketing Manager", "Head of Digital", "CMO"],
        growthAreas: ["Marketing automation platforms", "Data-driven marketing", "Video and influencer marketing", "Performance marketing optimization"],
        marketInsight: "Digital marketing offers diverse opportunities with continuous evolution in platforms and strategies."
    },
    "Agriculture & Agribusiness Careers": {
        overview: "Agriculture and Agribusiness professionals work in crop production, farm management, agricultural technology, and food supply chain management. This field combines traditional farming with modern technology and business practices.",
        keySkills: ["Crop and livestock management", "Agricultural science and technology", "Farm operations and equipment", "Sustainable farming practices", "Agricultural business management", "Supply chain and logistics", "Quality control and food safety"],
        jobRoles: ["Farm Worker", "Agricultural Technician", "Farm Manager", "Agronomist", "Agricultural Scientist", "Agribusiness Manager", "Agricultural Director"],
        growthAreas: ["Precision agriculture technology", "Sustainable and organic farming", "Agricultural data analytics", "Vertical farming and AgTech"],
        marketInsight: "Agriculture is evolving with technology adoption, creating new opportunities in AgTech and sustainable farming."
    },
    "Human Resources Careers": {
        overview: "Human Resources professionals manage employee relations, talent acquisition, benefits administration, and organizational development. HR ensures effective workforce management and positive workplace culture.",
        keySkills: ["Talent acquisition and recruiting", "Employee relations and conflict resolution", "HR policies and compliance", "Benefits and compensation administration", "Performance management", "HRIS systems (Workday, ADP)", "Training and development"],
        jobRoles: ["HR Assistant", "HR Coordinator", "Recruiter", "HR Generalist", "HR Business Partner", "HR Manager", "Talent Acquisition Manager", "CHRO"],
        growthAreas: ["HR analytics and data-driven HR", "Employee experience design", "HR technology implementation", "Organizational development"],
        marketInsight: "HR roles are essential across all organizations with growing emphasis on employee experience and analytics."
    },

    "Law & Legal Services Careers": {
        overview: "Legal professionals provide legal advice, represent clients, and ensure regulatory compliance. This includes lawyers, paralegals, legal researchers, and compliance specialists across various legal specializations.",
        keySkills: ["Legal research and writing", "Contract drafting and review", "Litigation and dispute resolution", "Regulatory compliance", "Client counseling and communication", "Legal document management", "Court procedures and rules"],
        jobRoles: ["Paralegal", "Legal Assistant", "Associate Attorney", "Compliance Officer", "Corporate Counsel", "Senior Attorney", "Partner", "General Counsel"],
        growthAreas: ["Legal technology adoption", "Specialized practice areas", "Alternative dispute resolution", "Compliance and regulatory expertise"],
        marketInsight: "Legal careers offer strong earning potential with specialization significantly impacting opportunities."
    },
    "Business Development Careers": {
        overview: "Business Development professionals identify growth opportunities, build strategic partnerships, and drive revenue expansion. This includes market research, partnership development, and strategic planning.",
        keySkills: ["Market research and opportunity identification", "Strategic partnership development", "Negotiation and deal-making", "Sales and relationship management", "Business strategy and planning", "Presentation and pitching", "CRM and pipeline management"],
        jobRoles: ["Business Development Representative", "BD Associate", "Business Development Manager", "Partnerships Manager", "Senior BD Manager", "Director of Business Development", "VP of Business Development"],
        growthAreas: ["Strategic partnerships and alliances", "International market expansion", "Corporate development and M&A", "Industry vertical specialization"],
        marketInsight: "Business development roles offer significant earning potential with commission and bonus structures."
    },

    "Culinary Arts Careers": {
        overview: "Culinary professionals create food experiences through cooking, menu development, and kitchen management. This includes line cooking, pastry arts, food service management, and culinary entrepreneurship.",
        keySkills: ["Cooking techniques and methods", "Menu planning and development", "Food safety and sanitation", "Kitchen operations management", "Cost control and inventory", "Team leadership in kitchen settings", "Creativity and presentation"],
        jobRoles: ["Line Cook", "Prep Cook", "Sous Chef", "Pastry Chef", "Head Chef", "Executive Chef", "Culinary Director", "Restaurant Owner"],
        growthAreas: ["Specialized cuisine expertise", "Food business management", "Sustainable and farm-to-table practices", "Culinary education and media"],
        marketInsight: "Culinary careers offer creative expression with opportunities in restaurants, hotels, and food entrepreneurship."
    },
    "Consulting & Advisory Careers": {
        overview: "Consultants provide expert advice to organizations on strategy, operations, technology, and specialized domains. This includes management consulting, IT consulting, and industry-specific advisory services.",
        keySkills: ["Problem-solving and analytical thinking", "Client relationship management", "Research and data analysis", "Presentation and communication", "Project management", "Industry and domain expertise", "Strategic thinking and recommendations"],
        jobRoles: ["Analyst", "Associate Consultant", "Consultant", "Senior Consultant", "Manager", "Senior Manager", "Principal", "Partner"],
        growthAreas: ["Industry specialization", "Digital transformation consulting", "Change management expertise", "Thought leadership development"],
        marketInsight: "Consulting offers exposure to diverse industries with clear progression and high earning potential."
    },
    "IT Support & Services Careers": {
        overview: "IT Support professionals provide technical assistance and maintain IT infrastructure for organizations. This includes help desk support, system administration, desktop support, and IT service management.",
        keySkills: ["Technical troubleshooting and problem-solving", "Help desk and ticketing systems", "Hardware and software support", "Operating systems (Windows, macOS, Linux)", "Active Directory and user management", "Network basics and connectivity", "Customer service and communication"],
        jobRoles: ["Help Desk Technician", "IT Support Specialist", "Desktop Support Analyst", "System Administrator", "IT Support Manager", "IT Service Manager", "IT Director"],
        growthAreas: ["Cloud certifications (AWS, Azure)", "Cybersecurity fundamentals", "ITIL and service management", "Automation and scripting"],
        marketInsight: "IT support offers stable career entry with clear paths to specialized technical or management roles."
    },
    "Public Relations & Communications Careers": {
        overview: "PR and Communications professionals manage organizational reputation, media relations, and strategic communications. This includes press relations, corporate communications, crisis management, and content strategy.",
        keySkills: ["Media relations and press releases", "Corporate communications and messaging", "Crisis communication management", "Content strategy and creation", "Stakeholder communication", "Social media and digital PR", "Event planning and management"],
        jobRoles: ["PR Assistant", "Communications Coordinator", "PR Specialist", "Communications Manager", "PR Manager", "Director of Communications", "VP of Communications", "Chief Communications Officer"],
        growthAreas: ["Digital PR and influencer relations", "Corporate social responsibility", "Executive communications", "Data-driven PR measurement"],
        marketInsight: "PR and communications roles are essential for brand management with growing digital opportunities."
    },
    "Aviation & Aerospace Careers": {
        overview: "Aviation and Aerospace professionals work in flight operations, aircraft maintenance, aerospace engineering, and aviation management. This field combines technical expertise with strict safety and regulatory compliance.",
        keySkills: ["Aviation regulations and compliance (FAA)", "Aircraft systems and operations", "Flight planning and navigation", "Aviation safety procedures", "Technical maintenance and inspection", "Air traffic procedures", "Documentation and record keeping"],
        jobRoles: ["Ground Operations Agent", "Aircraft Mechanic", "Flight Dispatcher", "Air Traffic Controller", "Pilot", "Aviation Manager", "Aerospace Engineer", "Director of Operations"],
        growthAreas: ["Advanced certifications and ratings", "Aerospace technology and drones", "Aviation management degrees", "Sustainable aviation initiatives"],
        marketInsight: "Aviation offers prestigious careers with strong job security and opportunities for global travel."
    }
};

const Learnmore = () => {
    const navigate = useNavigate();
    const {
        predictionResults,
        resumeText,
        learningRoadmapByPath,
        setLearningRoadmapByPath,
        certificationDataByPath,
        setCertificationDataByPath,
        skillsInsightsByPath,
        setSkillsInsightsByPath,
        historyRecordId,
    } = useDashboard();
    const { currentUser, logout, getToken } = useAuth();
    const [searchParams] = useSearchParams();
    const [showWhyOthersLower, setShowWhyOthersLower] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [menuOpen, setMenuOpen] = useState(false);
    const [showCareerPaths, setShowCareerPaths] = useState(false);
    const [skillsInsightsLoading, setSkillsInsightsLoading] = useState(false);
    const [skillsInsightsError, setSkillsInsightsError] = useState(null);

    // Right drawer state - default to closed
    const [activePanel, setActivePanel] = useState(null);

    // Handle panel toggle
    const handlePanelToggle = useCallback((panel) => {
        setActivePanel(panel);
    }, []);

    // Handle drawer close
    const handleDrawerClose = useCallback(() => {
        setActivePanel(null);
    }, []);

    // Ref for the logo element — captured by html2canvas for the PDF
    const logoRef = useRef(null);

    // Get extracted keywords — declared here so it's available in the handleExportPdf dep array below
    const extractedKeywords = predictionResults?.extracted_keywords || [];

    // Scroll to top when component mounts
    useEffect(() => {
        window.scrollTo(0, 0);
    }, []);

    // ── Auto-save learning roadmap to history when generated ─────────────────────
    useEffect(() => {
        if (!learningRoadmapByPath || !historyRecordId) return;
        const token = getToken();
        if (!token) return;
        apiService.updateHistory(token, historyRecordId, {
            learning_roadmap_by_path: learningRoadmapByPath,
        }).catch((err) => console.warn('History roadmap update failed (non-critical):', err));
    }, [learningRoadmapByPath, historyRecordId, getToken]);

    // ── Auto-save certification data to history when generated ─────────────────
    useEffect(() => {
        if (!certificationDataByPath || !historyRecordId) return;
        const token = getToken();
        if (!token) return;
        apiService.updateHistory(token, historyRecordId, {
            certification_data_by_path: certificationDataByPath,
        }).catch((err) => console.warn('History cert update failed (non-critical):', err));
    }, [certificationDataByPath, historyRecordId, getToken]);

    // ── Auto-save skills insights to history when generated ────────────────────
    useEffect(() => {
        if (!skillsInsightsByPath || !historyRecordId) return;
        const token = getToken();
        if (!token) return;
        apiService.updateHistory(token, historyRecordId, {
            skills_insights_by_path: skillsInsightsByPath,
        }).catch((err) => console.warn('History skills insights update failed (non-critical):', err));
    }, [skillsInsightsByPath, historyRecordId, getToken]);

    const topThree = predictionResults?.top_predictions?.slice(0, 3) || [];
    const requestedCareerPath = searchParams.get('career');
    const requestedCareerName = requestedCareerPath ? requestedCareerPath.trim() : null;
    const selectedIndex = requestedCareerName
        ? topThree.findIndex((item) => item.career_path === requestedCareerName)
        : 0;
    const resolvedIndex = selectedIndex >= 0 ? selectedIndex : 0;
    const selectedPrediction = topThree[resolvedIndex];
    const careerName = selectedPrediction?.career_path || 'This Career';
    const rawConfidence = selectedPrediction?.raw_confidence || 0;
    const selectedRank = selectedPrediction ? resolvedIndex + 1 : 1;

    const extractedKeywordsByPath = predictionResults?.extracted_keywords_by_path || {};
    const totalDistinctiveKeywordsByPath = predictionResults?.total_distinctive_keywords_by_path || {};
    const selectedExtractedKeywords = extractedKeywordsByPath[careerName] || extractedKeywords;
    const totalDistinctiveKeywords = totalDistinctiveKeywordsByPath[careerName]
        ?? predictionResults?.total_distinctive_keywords
        ?? selectedExtractedKeywords.length
        ?? extractedKeywords.length;

    // Calculate profile fit score — now imported from shared utility
    // calculateProfileFit is imported from '../../utils/profileFit'

    const selectedLearningRoadmap = learningRoadmapByPath?.[careerName] ?? null;
    const selectedCertificationData = certificationDataByPath?.[careerName] ?? null;
    const selectedSkillsInsights = skillsInsightsByPath?.[careerName] ?? null;

    // ── Auto-fetch skills insights when page loads for a career ────────────────
    useEffect(() => {
        if (!careerName || !resumeText) return;
        if (skillsInsightsByPath?.[careerName]) return; // Already cached
        let cancelled = false;
        setSkillsInsightsLoading(true);
        setSkillsInsightsError(null);
        apiService.generateSkillsInsights(careerName, resumeText)
            .then((data) => {
                if (cancelled) return;
                if (data?.insights) {
                    setSkillsInsightsByPath((prev) => ({
                        ...(prev || {}),
                        [careerName]: data.insights,
                    }));
                }
            })
            .catch((err) => {
                if (cancelled) return;
                console.error('Skills insights error:', err);
                setSkillsInsightsError(err.message || 'Failed to load');
            })
            .finally(() => {
                if (!cancelled) setSkillsInsightsLoading(false);
            });
        return () => { cancelled = true; };
    }, [careerName, resumeText, skillsInsightsByPath, setSkillsInsightsByPath]);

    // Export to PDF handler
    const handleExportPdf = useCallback(async () => {
        if (isExporting) return;
        setIsExporting(true);
        try {
            await exportToPdf({
                topThree: predictionResults?.top_predictions?.slice(0, 3) ?? [],
                calculateProfileFit,
                learningRoadmap: selectedLearningRoadmap,
                certificationData: selectedCertificationData,
                careerContent,
                logoRef,
                extractedKeywords: selectedExtractedKeywords,
                selectedCareerPath: careerName,
            });
        } finally {
            setIsExporting(false);
        }
    }, [isExporting, predictionResults, selectedLearningRoadmap, selectedCertificationData, selectedExtractedKeywords, calculateProfileFit, careerContent, logoRef]);

    const profileFitScore = React.useMemo(() => {
        const nFits = normalizeTop3Fits(topThree);
        return nFits[resolvedIndex] ?? calculateProfileFit(rawConfidence);
    }, [topThree, resolvedIndex, rawConfidence]);

    const nextMatchScore = React.useMemo(() => {
        const nFits = normalizeTop3Fits(topThree);
        return topThree[resolvedIndex + 1]
            ? (nFits[resolvedIndex + 1] ?? calculateProfileFit(topThree[resolvedIndex + 1].raw_confidence))
            : 0;
    }, [topThree, resolvedIndex]);

    // Calculate evidence breakdown factors (based on actual data)
    const keywordCount = totalDistinctiveKeywords;
    const scoreDifferential = Math.max(0, profileFitScore - nextMatchScore);

    // Dynamic factor weights based on actual data
    const resumeSignalStrength = Math.min(40, Math.round(keywordCount * 4));
    const skillAlignment = Math.min(35, Math.round(25 + (scoreDifferential / 2)));
    const experienceMatch = Math.min(25, Math.round(15 + (rawConfidence / 2)));
    const careerFit = 100 - resumeSignalStrength - skillAlignment - experienceMatch;

    // Get career content for the matched career
    const content = careerContent[careerName] || {
        overview: `${careerName} offers diverse opportunities for professionals with relevant skills and experience.`,
        keySkills: ["Industry-specific skills", "Technical proficiency", "Communication skills", "Problem-solving abilities"],
        jobRoles: ["Entry-level positions", "Mid-level roles", "Senior positions", "Leadership roles"],
        growthAreas: ["Advanced certifications", "Leadership development", "Technical specializations"],
        marketInsight: "This career offers competitive opportunities with potential for growth."
    };

    // Get career icon
    const getCareerIcon = (name) => {
        const Icon = careerIcons[name] || careerIcons["Software Development Careers"];
        return <Icon size={40} color="white" />;
    };

    // Generate why recommended explanation
    const generateWhyExplanation = () => {
        const reasons = [];

        reasons.push(`The AI analyzed your resume text and found vocabulary patterns that strongly match this career path.`);

        if (content.keySkills && content.keySkills.length > 0) {
            reasons.push(`Your experience aligns with core competencies like ${content.keySkills[0].split('(')[0].trim()} and ${content.keySkills[1]?.split('(')[0].trim() || 'related skills'}.`);
        }

        reasons.push(`Out of 26 career paths compared, this showed the strongest text pattern match with your background.`);

        return reasons;
    };

    // Generate why others ranked lower
    const generateComparisonExplanation = () => {
        const nFits = normalizeTop3Fits(topThree);
        return topThree
            .filter((_, index) => index !== resolvedIndex)
            .map((item, _, __, origIdx = topThree.indexOf(item)) => {
                const score = nFits[origIdx] ?? calculateProfileFit(item.raw_confidence);
                const gap = profileFitScore - score;
                const absGap = Math.abs(gap);
                const direction = gap >= 0 ? 'lower' : 'higher';
                const reason = gap >= 0
                    ? `${absGap}% lower due to fewer matching keywords and weaker experience signals.`
                    : `${absGap}% higher, indicating this path aligns more strongly with your current resume.`;

                return {
                    career: item.career_path,
                    score,
                    reason,
                    direction,
                };
            });
    };

    // If no prediction results, show message
    if (!selectedPrediction) {
        const handleLogout = () => {
            setMenuOpen(false);
            logout();
            navigate('/');
        };

        return (
            <div className="learnmore-container">
                <header className="learnmore-header">
                    <div className="header-content">
                        <div className="learnmore-header-left">
                            <h1 className="learnmore-brand" onClick={() => navigate('/')}>
                                <Logo variant="modern" />
                            </h1>
                        </div>

                        <nav className="learnmore-top-nav">
                            <button className="learnmore-nav-tab" onClick={() => navigate('/dashboard')}>Dashboard</button>
                            <button className="learnmore-nav-tab" onClick={() => setShowCareerPaths(true)}>Career Paths</button>
                            <button className="learnmore-nav-tab" onClick={() => navigate('/history')}>History</button>
                            <button className="learnmore-nav-tab active" type="button" disabled>Detailed</button>
                        </nav>

                        {/* Hamburger Button (mobile only) */}
                        <button
                            className={`hamburger-btn ${menuOpen ? 'open' : ''}`}
                            onClick={() => setMenuOpen(!menuOpen)}
                            aria-label="Toggle menu"
                        >
                            <span></span>
                            <span></span>
                            <span></span>
                        </button>

                        <div className="learnmore-header-right">
                            {currentUser && (
                                <div className="learnmore-profile-chip">
                                    <span className="learnmore-profile-dot" aria-hidden="true"></span>
                                    <span className="learnmore-greeting">{currentUser.username}</span>
                                </div>
                            )}
                            <div className="learnmore-action-group">
                                {currentUser?.is_admin && (
                                    <button className="learnmore-admin-btn" onClick={() => navigate('/admin')}>
                                        🛡 Admin
                                    </button>
                                )}
                                <button className="learnmore-logout-btn" onClick={handleLogout}>
                                    Logout
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Mobile nav drawer */}
                    {menuOpen && (
                        <div className="mobile-nav-drawer">
                            {currentUser && (
                                <span className="mobile-nav-greeting">Hello, {currentUser.username}</span>
                            )}
                            <button className="mobile-nav-btn" onClick={() => { navigate('/dashboard'); setMenuOpen(false); }}>
                                Dashboard
                            </button>
                            <button className="mobile-nav-btn" onClick={() => { setShowCareerPaths(true); setMenuOpen(false); }}>
                                Career Paths
                            </button>
                            <button className="mobile-nav-btn" onClick={() => { navigate('/history'); setMenuOpen(false); }}>
                                History
                            </button>
                            {currentUser?.is_admin && (
                                <button className="mobile-nav-btn" onClick={() => { navigate('/admin'); setMenuOpen(false); }}>
                                    🛡 Admin
                                </button>
                            )}
                            <button className="mobile-nav-btn mobile-nav-btn-logout" onClick={handleLogout}>
                                Logout
                            </button>
                        </div>
                    )}
                </header>
                <main className="learnmore-main">
                    <div className="learnmore-no-data">
                        <h2>No Career Analysis Available</h2>
                        <p>Please upload your resume on the Dashboard first to see career recommendations.</p>
                        <button className="learnmore-cta-btn" onClick={() => navigate('/dashboard')}>
                            Go to Dashboard
                        </button>
                    </div>
                </main>

                <CareerPathsModal
                    isOpen={showCareerPaths}
                    onClose={() => setShowCareerPaths(false)}
                />
            </div>
        );
    }

    const whyReasons = generateWhyExplanation();
    const comparisonItems = generateComparisonExplanation();
    const comparisonLabel = selectedRank === 1
        ? 'Why Other Career Paths Ranked Lower'
        : 'How This Career Compares';
    const comparisonHelper = selectedRank === 1
        ? 'These alternatives scored lower based on the same resume signals.'
        : 'These are your other top matches and how they compare to this path.';
    const setLearningRoadmapForPath = useCallback((roadmap) => {
        setLearningRoadmapByPath((prev) => {
            const next = { ...(prev || {}) };
            if (!roadmap) {
                delete next[careerName];
                return Object.keys(next).length ? next : null;
            }
            next[careerName] = roadmap;
            return next;
        });
    }, [careerName, setLearningRoadmapByPath]);
    const setCertificationDataForPath = useCallback((certs) => {
        setCertificationDataByPath((prev) => {
            const next = { ...(prev || {}) };
            if (!certs) {
                delete next[careerName];
                return Object.keys(next).length ? next : null;
            }
            next[careerName] = certs;
            return next;
        });
    }, [careerName, setCertificationDataByPath]);
    const handleLogout = () => {
        setMenuOpen(false);
        logout();
        navigate('/');
    };

    return (
        <div className="learnmore-container">
            {/* Header */}
            <header className="learnmore-header">
                <div className="header-content">
                    <div className="learnmore-header-left">
                        <h1 ref={logoRef} className="learnmore-brand" onClick={() => navigate('/')}>
                            <Logo variant="modern" />
                        </h1>
                    </div>

                    <nav className="learnmore-top-nav">
                        <button className="learnmore-nav-tab" onClick={() => navigate('/dashboard')}>Dashboard</button>
                        <button className="learnmore-nav-tab" onClick={() => setShowCareerPaths(true)}>Career Paths</button>
                        <button className="learnmore-nav-tab" onClick={() => navigate('/history')}>History</button>
                        <button className="learnmore-nav-tab active" type="button" disabled>Detailed</button>
                    </nav>

                    {/* Hamburger Button (mobile only) */}
                    <button
                        className={`hamburger-btn ${menuOpen ? 'open' : ''}`}
                        onClick={() => setMenuOpen(!menuOpen)}
                        aria-label="Toggle menu"
                    >
                        <span></span>
                        <span></span>
                        <span></span>
                    </button>

                    <div className="learnmore-header-right">
                        {currentUser && (
                            <div className="learnmore-profile-chip">
                                <span className="learnmore-profile-dot" aria-hidden="true"></span>
                                <span className="learnmore-greeting">{currentUser.username}</span>
                            </div>
                        )}
                        <div className="learnmore-action-group">
                            <button className="learnmore-export-btn" onClick={handleExportPdf} disabled={isExporting}>
                                {React.createElement(otherIcons[isExporting ? "FaSpinner" : "FaDownload"], { size: 13 })}
                                <span>{isExporting ? 'Exporting...' : 'Export PDF'}</span>
                            </button>
                            {currentUser?.is_admin && (
                                <button className="learnmore-admin-btn" onClick={() => navigate('/admin')}>
                                    🛡 Admin
                                </button>
                            )}
                            <button className="learnmore-logout-btn" onClick={handleLogout}>
                                Logout
                            </button>
                        </div>
                    </div>
                </div>

                {/* Mobile nav drawer */}
                {menuOpen && (
                    <div className="mobile-nav-drawer">
                        {currentUser && (
                            <span className="mobile-nav-greeting">Hello, {currentUser.username}</span>
                        )}
                        <button className="mobile-nav-btn" onClick={() => { navigate('/dashboard'); setMenuOpen(false); }}>
                            Dashboard
                        </button>
                        <button className="mobile-nav-btn" onClick={() => { setShowCareerPaths(true); setMenuOpen(false); }}>
                            Career Paths
                        </button>
                        <button className="mobile-nav-btn" onClick={() => { navigate('/history'); setMenuOpen(false); }}>
                            History
                        </button>
                        <button className="mobile-nav-btn" onClick={() => { handleExportPdf(); setMenuOpen(false); }} disabled={isExporting}>
                            {React.createElement(otherIcons[isExporting ? "FaSpinner" : "FaDownload"], { size: 13 })}
                            <span>{isExporting ? 'Exporting...' : 'Export PDF'}</span>
                        </button>
                        {currentUser?.is_admin && (
                            <button className="mobile-nav-btn" onClick={() => { navigate('/admin'); setMenuOpen(false); }}>
                                🛡 Admin
                            </button>
                        )}
                        <button className="mobile-nav-btn mobile-nav-btn-logout" onClick={handleLogout}>
                            Logout
                        </button>
                    </div>
                )}
            </header>

            {/* Main Content */}
            <main className="learnmore-main">
                <div className="learnmore-content">

                    {/* SECTION 1: Hero Card */}
                    <div className="hero-card">
                        <div className="hero-card-main">
                            <div className="hero-card-left">
                                <div className="hero-icon">
                                    {getCareerIcon(careerName)}
                                </div>
                                <div className="hero-info">
                                    <span className="hero-label">#{selectedRank} Career Match</span>
                                    <h1 className="hero-title">{careerName}</h1>
                                    <p className="hero-insight">{content.marketInsight}</p>
                                </div>
                            </div>
                            <div className="hero-meta">
                                <span className="hero-meta-pill">
                                    {selectedRank === 1 ? 'Top Recommendation' : 'Alternate Recommendation'}
                                </span>
                                <span className="hero-meta-text">Compared across 26 career paths</span>
                            </div>
                        </div>

                        <div className="hero-score">
                            <span className="hero-score-value">{profileFitScore}%</span>
                            <span className="hero-score-label">Profile Fit</span>
                            <div className="hero-score-track" aria-hidden="true">
                                <div className="hero-score-fill" style={{ width: `${profileFitScore}%` }}></div>
                            </div>
                        </div>
                    </div>

                    {/* SECTION 2: Why This Career Is Recommended */}
                    <div className="why-section">
                        <h3 className="why-title">Why This Career Path Is Recommended</h3>
                        <p className="why-subtitle">Here's how the AI interpreted your resume</p>
                        <ul className="why-list">
                            {whyReasons.map((reason, index) => (
                                <li key={index} className="why-item">
                                    <span className="why-bullet">{React.createElement(otherIcons["FaArrowRight"], { size: 12, color: "#10b981" })}</span>
                                    {reason}
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* SECTION 3: How Match Was Calculated */}
                    <div className="evidence-section">
                        <h3 className="evidence-title">How Your Match Was Calculated</h3>
                        <div className="evidence-explanation">
                            <p>Your match is based on analyzing text patterns in your resume against career-specific vocabulary from our training data.</p>
                            <div className="evidence-stats">
                                <div className="evidence-stat">
                                    <span className="evidence-stat-value">{keywordCount}</span>
                                    <span className="evidence-stat-label">Distinctive keywords detected</span>
                                </div>
                                <div className="evidence-stat">
                                    <span className="evidence-stat-value">26</span>
                                    <span className="evidence-stat-label">Career paths compared</span>
                                </div>
                                <div className="evidence-stat">
                                    <span className="evidence-stat-value">+{scoreDifferential}%</span>
                                    <span className="evidence-stat-label">Ahead of next match</span>
                                </div>
                            </div>
                            <p className="evidence-note">The model uses TF-IDF text analysis combined with Naive Bayes classification to identify which career vocabulary best matches your resume.</p>
                        </div>
                    </div>

                    {/* SECTION 4: Three-Column Summary */}
                    <div className="summary-grid">
                        {/* Column 1: Resume Signals */}
                        <div className="summary-card">
                            <h3 className="summary-card-title">
                                <span className="summary-icon">{React.createElement(otherIcons["FaMapMarkerAlt"], { size: 15, color: "#2563eb" })}</span>
                                Resume Signals Detected
                            </h3>
                            <p className="summary-card-desc">These keywords influenced your match</p>
                            <div className="keyword-list">
                                {selectedExtractedKeywords.length > 0 ? (
                                    selectedExtractedKeywords.slice(0, 6).map((item, index) => (
                                        <span key={index} className="keyword-chip">{item.keyword}</span>
                                    ))
                                ) : (
                                    <span className="no-keywords">Resume analyzed for patterns</span>
                                )}
                            </div>
                        </div>

                        {/* Column 2: Key Skills */}
                        <div className="summary-card">
                            <h3 className="summary-card-title">
                                <span className="summary-icon">{React.createElement(otherIcons["FaCheckCircle"], { size: 15, color: "#10b981" })}</span>
                                Skills Driving Your Score
                            </h3>
                            <p className="summary-card-desc">{selectedSkillsInsights ? 'Personalized to your resume' : 'Core competencies for this career'}</p>
                            <div className="skill-list-compact">
                                {skillsInsightsLoading ? (
                                    <>  
                                        {[1, 2, 3, 4].map((i) => (
                                            <div key={i} className="skill-item-compact skeleton-item">
                                                <span className="skeleton-line skeleton-line-long"></span>
                                            </div>
                                        ))}
                                    </>
                                ) : selectedSkillsInsights?.skills_driving_score ? (
                                    selectedSkillsInsights.skills_driving_score.slice(0, 4).map((skill, index) => (
                                        <div key={index} className="skill-item-compact">
                                            <span className="skill-item-icon">{React.createElement(otherIcons["FaCheck"], { size: 10, color: "#10b981" })}</span>
                                            {skill}
                                        </div>
                                    ))
                                ) : (
                                    content.keySkills.slice(0, 4).map((skill, index) => (
                                        <div key={index} className="skill-item-compact">
                                            <span className="skill-item-icon">{React.createElement(otherIcons["FaCheck"], { size: 10, color: "#10b981" })}</span>
                                            {skill}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>

                        {/* Column 3: Growth Areas */}
                        <div className="summary-card">
                            <h3 className="summary-card-title">
                                <span className="summary-icon">{React.createElement(otherIcons["FaRocket"], { size: 15, color: "#f59e0b" })}</span>
                                Improve Your Match
                            </h3>
                            <p className="summary-card-desc">{selectedSkillsInsights ? 'Based on gaps found in your resume' : 'Developing these can boost your score'}</p>
                            <div className="growth-list-compact">
                                {skillsInsightsLoading ? (
                                    <>
                                        {[1, 2, 3, 4].map((i) => (
                                            <div key={i} className="growth-item-compact skeleton-item">
                                                <span className="skeleton-line skeleton-line-long"></span>
                                            </div>
                                        ))}
                                    </>
                                ) : selectedSkillsInsights?.improve_your_match ? (
                                    selectedSkillsInsights.improve_your_match.slice(0, 4).map((area, index) => (
                                        <div key={index} className="growth-item-compact">
                                            <span className="growth-item-icon">{React.createElement(otherIcons["FaPlus"], { size: 9, color: "#f59e0b" })}</span>
                                            {area}
                                        </div>
                                    ))
                                ) : (
                                    content.growthAreas.slice(0, 4).map((area, index) => (
                                        <div key={index} className="growth-item-compact">
                                            <span className="growth-item-icon">{React.createElement(otherIcons["FaPlus"], { size: 9, color: "#f59e0b" })}</span>
                                            {area}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>

                    {/* SECTION 5: Career Comparison */}
                    <div className="comparison-section">
                        <h3 className="comparison-title">Your Top Career Matches</h3>
                        <p className="comparison-subtitle">Compared against 26 career paths</p>
                        <div className="comparison-bars">
                            {(() => {
                                const nFits = normalizeTop3Fits(topThree);
                                return topThree.map((career, index) => (
                                    <div key={index} className={`comparison-row ${index === resolvedIndex ? 'top' : ''}`}>
                                        <div className="comparison-rank">#{index + 1}</div>
                                        <div className="comparison-name">{career.career_path}</div>
                                        <div className="comparison-bar-wrap">
                                            <div
                                                className="comparison-bar-fill"
                                                style={{ width: `${nFits[index] ?? calculateProfileFit(career.raw_confidence)}%` }}
                                            />
                                        </div>
                                        <div className="comparison-percent">{nFits[index] ?? calculateProfileFit(career.raw_confidence)}%</div>
                                    </div>
                                ));
                            })()}
                        </div>

                        {/* Why Others Ranked Lower - Collapsible */}
                        {comparisonItems.length > 0 && (
                            <div className="why-others-section">
                                <button
                                    className="why-others-toggle"
                                    onClick={() => setShowWhyOthersLower(!showWhyOthersLower)}
                                >
                                    <span className="why-others-chevron">
                                        {showWhyOthersLower
                                            ? React.createElement(otherIcons["FaChevronUp"], { size: 10, color: "#64748b" })
                                            : React.createElement(otherIcons["FaChevronDown"], { size: 10, color: "#64748b" })
                                        }
                                    </span>
                                    {comparisonLabel}
                                </button>
                                {showWhyOthersLower && (
                                    <div className="why-others-content">
                                        <p className="why-others-helper">{comparisonHelper}</p>
                                        {comparisonItems.map((item, index) => (
                                            <div key={index} className="why-others-item">
                                                <div className="why-others-career">
                                                    <strong>{item.career}</strong>
                                                    <span className="why-others-score">{item.score}%</span>
                                                </div>
                                                <p className="why-others-reason">{item.reason}</p>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* SECTION 6: Job Roles */}
                    <div className="job-roles-section">
                        <h3 className="job-roles-title">Job Roles in This Career Path</h3>
                        <p className="job-roles-subtitle">Positions you could pursue with this background</p>
                        <div className="job-roles-grid">
                            {content.jobRoles.map((role, index) => (
                                <div key={index} className="job-role-card">
                                    <span className="job-role-level">{index < 2 ? 'Entry' : index < 4 ? 'Mid' : index < 6 ? 'Senior' : 'Lead'}</span>
                                    <span className="job-role-name">{role}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Important Note */}
                    <div className="note-section">
                        <p className="note-text">
                            <strong style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>{React.createElement(otherIcons["FaExclamationTriangle"], { size: 14, color: "#f59e0b" })} Note:</strong> This recommendation is based on text pattern analysis of your resume.
                            It's a starting point for exploration, not a definitive career assessment.
                            The AI cannot evaluate soft skills, personal preferences, or cultural fit.
                        </p>
                    </div>

                </div>
            </main>

            {/* Right Side Dock and Drawer */}
            <RightDock
                activePanel={activePanel}
                onPanelToggle={handlePanelToggle}
            />

            <RightDrawer
                isOpen={activePanel !== null}
                activePanel={activePanel || 'jobs'}
                onClose={handleDrawerClose}
            >
                {activePanel === 'jobs' && (
                    <JobsPanel
                        careerPath={careerName}
                        jobRoles={content.jobRoles}
                    />
                )}
                {activePanel === 'learning' && (
                    <LearningPanel
                        careerPath={careerName}
                        growthAreas={content.growthAreas}
                        resumeText={resumeText}
                        learningRoadmap={selectedLearningRoadmap}
                        setLearningRoadmap={setLearningRoadmapForPath}
                    />
                )}
                {activePanel === 'certifications' && (
                    <CertificationPanel
                        careerPath={careerName}
                        growthAreas={content.growthAreas}
                        resumeText={resumeText}
                        certificationData={selectedCertificationData}
                        setCertificationData={setCertificationDataForPath}
                    />
                )}
            </RightDrawer>

            <CareerPathsModal
                isOpen={showCareerPaths}
                onClose={() => setShowCareerPaths(false)}
            />

        </div>
    );
};

export default Learnmore;
