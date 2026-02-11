import React, { useEffect, useRef, useCallback } from 'react';
import { FaTimes, FaBriefcase, FaBook, FaCertificate } from 'react-icons/fa';
import './RightDrawer.css';

/**
 * RightDrawer - A sliding drawer panel from the right side of the screen
 * 
 * @param {Object} props
 * @param {boolean} props.isOpen - Whether the drawer is open
 * @param {string} props.activePanel - The active panel type ('jobs' | 'learning' | 'certifications')
 * @param {Function} props.onClose - Callback to close the drawer
 * @param {React.ReactNode} props.children - Content to render inside the drawer
 */
const RightDrawer = ({ isOpen, activePanel, onClose, children }) => {
    const drawerRef = useRef(null);
    const closeButtonRef = useRef(null);
    const previousActiveElement = useRef(null);

    // Panel configurations
    const panelConfig = {
        jobs: {
            title: 'Job Opportunities',
            icon: FaBriefcase,
            id: 'jobs-drawer'
        },
        learning: {
            title: 'Learning Materials',
            icon: FaBook,
            id: 'learning-drawer'
        },
        certifications: {
            title: 'Certifications',
            icon: FaCertificate,
            id: 'certifications-drawer'
        }
    };

    const currentPanel = panelConfig[activePanel] || panelConfig.jobs;
    const IconComponent = currentPanel.icon;

    // Handle ESC key to close drawer
    const handleKeyDown = useCallback((event) => {
        if (event.key === 'Escape' && isOpen) {
            onClose();
        }
    }, [isOpen, onClose]);

    // Focus trap implementation
    const handleTabKey = useCallback((event) => {
        if (!isOpen || !drawerRef.current) return;

        const focusableElements = drawerRef.current.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const firstFocusable = focusableElements[0];
        const lastFocusable = focusableElements[focusableElements.length - 1];

        if (event.key === 'Tab') {
            if (event.shiftKey) {
                if (document.activeElement === firstFocusable) {
                    event.preventDefault();
                    lastFocusable?.focus();
                }
            } else {
                if (document.activeElement === lastFocusable) {
                    event.preventDefault();
                    firstFocusable?.focus();
                }
            }
        }
    }, [isOpen]);

    // Set up event listeners
    useEffect(() => {
        document.addEventListener('keydown', handleKeyDown);
        document.addEventListener('keydown', handleTabKey);

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.removeEventListener('keydown', handleTabKey);
        };
    }, [handleKeyDown, handleTabKey]);

    // Manage focus when drawer opens/closes
    useEffect(() => {
        if (isOpen) {
            // Store the currently focused element
            previousActiveElement.current = document.activeElement;
            // Focus the close button when drawer opens
            setTimeout(() => {
                closeButtonRef.current?.focus();
            }, 100);
            // Prevent body scroll
            document.body.style.overflow = 'hidden';
        } else {
            // Restore focus when drawer closes
            document.body.style.overflow = '';
            if (previousActiveElement.current) {
                previousActiveElement.current.focus();
            }
        }

        return () => {
            document.body.style.overflow = '';
        };
    }, [isOpen]);

    // Handle backdrop click
    const handleBackdropClick = (event) => {
        if (event.target === event.currentTarget) {
            onClose();
        }
    };

    return (
        <>
            {/* Backdrop overlay */}
            <div
                className={`drawer-backdrop ${isOpen ? 'drawer-backdrop--visible' : ''}`}
                onClick={handleBackdropClick}
                aria-hidden="true"
            />

            {/* Drawer panel */}
            <div
                ref={drawerRef}
                id={currentPanel.id}
                className={`right-drawer ${isOpen ? 'right-drawer--open' : ''}`}
                role="dialog"
                aria-modal="true"
                aria-labelledby="drawer-title"
                aria-hidden={!isOpen}
            >
                {/* Drawer Header */}
                <div className="drawer-header">
                    <div className="drawer-header-title">
                        <IconComponent className="drawer-header-icon" />
                        <h2 id="drawer-title" className="drawer-title">
                            {currentPanel.title}
                        </h2>
                    </div>
                    <button
                        ref={closeButtonRef}
                        className="drawer-close-btn"
                        onClick={onClose}
                        aria-label="Close panel"
                    >
                        <FaTimes />
                    </button>
                </div>

                {/* Drawer Content */}
                <div className="drawer-content">
                    {children}
                </div>
            </div>
        </>
    );
};

export default RightDrawer;
