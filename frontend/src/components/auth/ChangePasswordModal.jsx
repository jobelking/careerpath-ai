import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { authIcons } from '../../utils/authIcons';
import './ChangePasswordModal.css';

const ChangePasswordModal = ({ isOpen, onClose }) => {
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showCurrentPassword, setShowCurrentPassword] = useState(false);
    const [showNewPassword, setShowNewPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const { changePassword } = useAuth();

    if (!isOpen) return null;

    const resetForm = () => {
        setCurrentPassword('');
        setNewPassword('');
        setConfirmPassword('');
        setShowCurrentPassword(false);
        setShowNewPassword(false);
        setShowConfirmPassword(false);
        setError('');
        setSuccess('');
    };

    const handleClose = () => {
        resetForm();
        onClose();
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        if (!currentPassword) {
            setError('Current password is required.');
            return;
        }
        if (!newPassword) {
            setError('New password is required.');
            return;
        }
        if (newPassword.length < 6) {
            setError('New password must be at least 6 characters.');
            return;
        }
        if (!confirmPassword) {
            setError('Please confirm your new password.');
            return;
        }
        if (newPassword !== confirmPassword) {
            setError('New passwords do not match.');
            return;
        }
        if (currentPassword === newPassword) {
            setError('New password must be different from your current password.');
            return;
        }

        setIsLoading(true);
        try {
            const result = await changePassword(currentPassword, newPassword);
            setSuccess(result.message || 'Password changed successfully!');
            setCurrentPassword('');
            setNewPassword('');
            setConfirmPassword('');
            // Auto-close after success
            setTimeout(() => handleClose(), 2000);
        } catch (err) {
            setError(err.message || 'Failed to change password.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleBackdrop = (e) => {
        if (e.target.className === 'cpw-modal-overlay') handleClose();
    };

    const EyeCurrentIcon = showCurrentPassword ? authIcons['FaEyeSlash'] : authIcons['FaEye'];
    const EyeNewIcon = showNewPassword ? authIcons['FaEyeSlash'] : authIcons['FaEye'];
    const EyeConfirmIcon = showConfirmPassword ? authIcons['FaEyeSlash'] : authIcons['FaEye'];

    return (
        <div className="cpw-modal-overlay" onClick={handleBackdrop}>
            <div className="cpw-modal-content">
                <button className="cpw-close-btn" onClick={handleClose} aria-label="Close">
                    {React.createElement(authIcons['FaTimes'], { size: 20 })}
                </button>

                <div className="cpw-header">
                    <div className="cpw-icon-wrap">
                        {React.createElement(authIcons['FaShieldAlt'], { className: 'cpw-shield-icon' })}
                    </div>
                    <h2>Change Password</h2>
                    <p>Update your account password</p>
                </div>

                {error && <div className="cpw-error">{error}</div>}
                {success && <div className="cpw-success">{success}</div>}

                <form onSubmit={handleSubmit} className="cpw-form" noValidate>
                    {/* Current Password */}
                    <div className="cpw-field">
                        <label htmlFor="cpw-current">Current Password</label>
                        <div className="cpw-input-wrap">
                            {React.createElement(authIcons['FaLock'], { className: 'cpw-input-icon' })}
                            <input
                                type={showCurrentPassword ? 'text' : 'password'}
                                id="cpw-current"
                                value={currentPassword}
                                onChange={(e) => setCurrentPassword(e.target.value)}
                                placeholder="Enter current password"
                                disabled={isLoading || !!success}
                            />
                            <button
                                type="button"
                                className="cpw-eye-toggle"
                                onClick={() => setShowCurrentPassword(v => !v)}
                                aria-label={showCurrentPassword ? 'Hide' : 'Show'}
                            >
                                {React.createElement(EyeCurrentIcon, { size: 16 })}
                            </button>
                        </div>
                    </div>

                    {/* New Password */}
                    <div className="cpw-field">
                        <label htmlFor="cpw-new">New Password</label>
                        <div className="cpw-input-wrap">
                            {React.createElement(authIcons['FaKey'], { className: 'cpw-input-icon' })}
                            <input
                                type={showNewPassword ? 'text' : 'password'}
                                id="cpw-new"
                                value={newPassword}
                                onChange={(e) => setNewPassword(e.target.value)}
                                placeholder="At least 6 characters"
                                disabled={isLoading || !!success}
                            />
                            <button
                                type="button"
                                className="cpw-eye-toggle"
                                onClick={() => setShowNewPassword(v => !v)}
                                aria-label={showNewPassword ? 'Hide' : 'Show'}
                            >
                                {React.createElement(EyeNewIcon, { size: 16 })}
                            </button>
                        </div>
                    </div>

                    {/* Confirm New Password */}
                    <div className="cpw-field">
                        <label htmlFor="cpw-confirm">Confirm New Password</label>
                        <div className="cpw-input-wrap">
                            {React.createElement(authIcons['FaKey'], { className: 'cpw-input-icon' })}
                            <input
                                type={showConfirmPassword ? 'text' : 'password'}
                                id="cpw-confirm"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                placeholder="Re-enter new password"
                                disabled={isLoading || !!success}
                            />
                            <button
                                type="button"
                                className="cpw-eye-toggle"
                                onClick={() => setShowConfirmPassword(v => !v)}
                                aria-label={showConfirmPassword ? 'Hide' : 'Show'}
                            >
                                {React.createElement(EyeConfirmIcon, { size: 16 })}
                            </button>
                        </div>
                    </div>

                    <button
                        type="submit"
                        className={`cpw-submit-btn ${isLoading ? 'loading' : ''} ${success ? 'success' : ''}`}
                        disabled={isLoading || !!success}
                    >
                        {success ? '✓ Password Changed' : isLoading ? 'Changing...' : 'Change Password'}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChangePasswordModal;
