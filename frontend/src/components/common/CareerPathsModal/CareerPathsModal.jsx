import React, { useEffect, useRef, useCallback } from 'react';
import { FaTimes, FaLaptopCode, FaChartLine, FaPalette, FaHeartbeat, FaGlobe } from 'react-icons/fa';
import { careerIcons } from '../../../utils/careerIcons';
import './CareerPathsModal.css';

const CAREER_CATEGORIES = [
    {
        label: 'Technology & Engineering',
        Icon: FaLaptopCode,
        paths: [
            'Software Development Careers',
            'Mobile Development Careers',
            'Data Science & AI Careers',
            'DevOps & Site Reliability Careers',
            'Network Administration Careers',
            'Cybersecurity Careers',
            'Quality Assurance & Testing Careers',
            'IT Support & Services Careers',
            'Engineering Careers',
        ],
    },
    {
        label: 'Business & Finance',
        Icon: FaChartLine,
        paths: [
            'Business Analysis Careers',
            'Business Development Careers',
            'Sales Careers',
            'Consulting & Advisory Careers',
            'Human Resources Careers',
            'Finance & Accounting Careers',
        ],
    },
    {
        label: 'Creative & Media',
        Icon: FaPalette,
        paths: [
            'Design & Creative Careers',
            'Digital Media & Marketing Careers',
            'Public Relations & Communications Careers',
        ],
    },
    {
        label: 'Health & Education',
        Icon: FaHeartbeat,
        paths: [
            'Healthcare Careers',
            'Fitness & Wellness Careers',
            'Education & Teaching Careers',
        ],
    },
    {
        label: 'Other Industries',
        Icon: FaGlobe,
        paths: [
            'Culinary Arts Careers',
            'Construction Careers',
            'Agriculture & Agribusiness Careers',
            'Aviation & Aerospace Careers',
            'Law & Legal Services Careers',
        ],
    },
];

/**
 * CareerPathsModal
 * Displays all 26 career paths the system can predict, grouped by category.
 */
const CareerPathsModal = ({ isOpen = true, onClose }) => {
    const modalRef = useRef(null);
    const closeButtonRef = useRef(null);
    const previousActiveElement = useRef(null);

    // Close on Escape
    const handleKeyDown = useCallback(
        (e) => {
            if (e.key === 'Escape') onClose();
        },
        [onClose]
    );

    useEffect(() => {
        if (!isOpen) {
            return undefined;
        }

        previousActiveElement.current = document.activeElement;
        document.addEventListener('keydown', handleKeyDown);
        document.body.style.overflow = 'hidden';
        setTimeout(() => closeButtonRef.current?.focus(), 100);

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = '';
            previousActiveElement.current?.focus();
        };
    }, [handleKeyDown, isOpen]);

    const handleBackdropClick = (e) => {
        if (e.target === e.currentTarget) onClose();
    };

    if (!isOpen) {
        return null;
    }

    const totalPaths = CAREER_CATEGORIES.reduce((sum, cat) => sum + cat.paths.length, 0);

    return (
        <div
            className="cp-modal-backdrop"
            onClick={handleBackdropClick}
            aria-modal="true"
            role="dialog"
            aria-labelledby="cp-modal-title"
        >
            <div className="cp-modal" ref={modalRef}>
                {/* Header */}
                <div className="cp-modal-header">
                    <div className="cp-modal-title-area">
                        <span className="cp-modal-badge">{totalPaths} Paths</span>
                        <h2 id="cp-modal-title" className="cp-modal-title">
                            Career Paths We Predict
                        </h2>
                        <p className="cp-modal-subtitle">
                            Our AI model analyses your resume against these career tracks to find your best match.
                        </p>
                    </div>
                    <button
                        ref={closeButtonRef}
                        className="cp-modal-close"
                        onClick={onClose}
                        aria-label="Close career paths"
                    >
                        <FaTimes />
                    </button>
                </div>

                {/* Content */}
                <div className="cp-modal-body">
                    {CAREER_CATEGORIES.map((category) => (
                        <div key={category.label} className="cp-category">
                            <div className="cp-category-header">
                                <span className="cp-category-icon"><category.Icon size={15} /></span>
                                <h3 className="cp-category-label">{category.label}</h3>
                                <span className="cp-category-count">{category.paths.length}</span>
                            </div>
                            <div className="cp-paths-grid">
                                {category.paths.map((path) => {
                                    const Icon = careerIcons[path];
                                    return (
                                        <div key={path} className="cp-path-card">
                                            <div className="cp-path-icon">
                                                {Icon && <Icon size={22} />}
                                            </div>
                                            <span className="cp-path-name">{path}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default CareerPathsModal;
