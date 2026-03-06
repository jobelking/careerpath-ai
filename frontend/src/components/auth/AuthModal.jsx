import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import apiService from '../../services/api/apiService';
import Logo from '../common/Logo';
import { authIcons } from '../../utils/authIcons';
import './AuthModal.css';

const OTP_LENGTH = 6;
const OTP_EXPIRY_SECONDS = 5 * 60; // 5 minutes
const RESEND_COOLDOWN_SECONDS = 60;

// ─── OTP View ────────────────────────────────────────────────────────────────
const OTPView = ({ email, onSuccess, onBack }) => {
    const [digits, setDigits] = useState(Array(OTP_LENGTH).fill(''));
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [expiryLeft, setExpiryLeft] = useState(OTP_EXPIRY_SECONDS);
    const [resendCooldown, setResendCooldown] = useState(RESEND_COOLDOWN_SECONDS);
    const [resendLoading, setResendLoading] = useState(false);
    const [resendSuccess, setResendSuccess] = useState('');
    const inputRefs = useRef([]);

    // Countdown timers
    useEffect(() => {
        const expiryTimer = setInterval(() => {
            setExpiryLeft(prev => Math.max(0, prev - 1));
        }, 1000);
        return () => clearInterval(expiryTimer);
    }, []);

    useEffect(() => {
        if (resendCooldown <= 0) return;
        const cooldownTimer = setInterval(() => {
            setResendCooldown(prev => Math.max(0, prev - 1));
        }, 1000);
        return () => clearInterval(cooldownTimer);
    }, [resendCooldown]);

    const formatTime = (seconds) => {
        const m = Math.floor(seconds / 60).toString().padStart(2, '0');
        const s = (seconds % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
    };

    const handleDigitChange = (index, value) => {
        // Accept only digits
        const cleaned = value.replace(/\D/g, '').slice(-1);
        const newDigits = [...digits];
        newDigits[index] = cleaned;
        setDigits(newDigits);
        setError('');

        // Auto-advance to next box
        if (cleaned && index < OTP_LENGTH - 1) {
            inputRefs.current[index + 1]?.focus();
        }
    };

    const handleKeyDown = (index, e) => {
        if (e.key === 'Backspace' && !digits[index] && index > 0) {
            inputRefs.current[index - 1]?.focus();
        }
        if (e.key === 'ArrowLeft' && index > 0) {
            inputRefs.current[index - 1]?.focus();
        }
        if (e.key === 'ArrowRight' && index < OTP_LENGTH - 1) {
            inputRefs.current[index + 1]?.focus();
        }
    };

    const handlePaste = (e) => {
        e.preventDefault();
        const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, OTP_LENGTH);
        const newDigits = [...digits];
        for (let i = 0; i < pasted.length; i++) {
            newDigits[i] = pasted[i];
        }
        setDigits(newDigits);
        // Focus last filled box
        const lastIndex = Math.min(pasted.length, OTP_LENGTH - 1);
        inputRefs.current[lastIndex]?.focus();
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const code = digits.join('');
        if (code.length < OTP_LENGTH) {
            setError('Please enter all 6 digits.');
            return;
        }
        if (expiryLeft <= 0) {
            setError('The code has expired. Please request a new one.');
            return;
        }

        setIsLoading(true);
        setError('');
        try {
            const data = await apiService.verifyOTP(email, code);
            onSuccess(data);
        } catch (err) {
            setError(err.message || 'Verification failed. Please try again.');
            // Clear boxes on error
            setDigits(Array(OTP_LENGTH).fill(''));
            inputRefs.current[0]?.focus();
        } finally {
            setIsLoading(false);
        }
    };

    const handleResend = async () => {
        if (resendCooldown > 0 || resendLoading) return;
        setResendLoading(true);
        setResendSuccess('');
        setError('');
        try {
            await apiService.resendOTP(email);
            setResendCooldown(RESEND_COOLDOWN_SECONDS);
            setExpiryLeft(OTP_EXPIRY_SECONDS);
            setDigits(Array(OTP_LENGTH).fill(''));
            setResendSuccess('A new code has been sent to your email.');
            inputRefs.current[0]?.focus();
        } catch (err) {
            setError(err.message || 'Failed to resend code. Please try again.');
        } finally {
            setResendLoading(false);
        }
    };

    const maskedEmail = email.replace(/(.{2})(.*)(?=@)/, (_, a, b) => a + '*'.repeat(b.length));

    return (
        <div className="otp-view">
            <div className="otp-icon-wrap">
                {React.createElement(authIcons['FaEnvelope'], { className: 'otp-envelope-icon' })}
            </div>
            <h2 className="otp-title">Check your email</h2>
            <p className="otp-subtitle">
                We sent a 6-digit verification code to<br />
                <strong>{maskedEmail}</strong>
            </p>

            {error && <div className="auth-error">{error}</div>}
            {resendSuccess && <div className="otp-success-msg">{resendSuccess}</div>}

            <form onSubmit={handleSubmit} className="otp-form" noValidate>
                <div className="otp-boxes" onPaste={handlePaste}>
                    {digits.map((digit, i) => (
                        <input
                            key={i}
                            ref={el => inputRefs.current[i] = el}
                            id={`otp-digit-${i}`}
                            type="text"
                            inputMode="numeric"
                            maxLength={1}
                            value={digit}
                            onChange={e => handleDigitChange(i, e.target.value)}
                            onKeyDown={e => handleKeyDown(i, e)}
                            className={`otp-box ${digit ? 'otp-box--filled' : ''} ${error ? 'otp-box--error' : ''}`}
                            autoFocus={i === 0}
                            autoComplete="one-time-code"
                        />
                    ))}
                </div>

                {/* Expiry countdown */}
                <div className={`otp-expiry ${expiryLeft < 60 ? 'otp-expiry--urgent' : ''}`}>
                    {expiryLeft > 0
                        ? <>Code expires in <strong>{formatTime(expiryLeft)}</strong></>
                        : <span className="otp-expired">Code expired — please resend</span>
                    }
                </div>

                <button
                    type="submit"
                    className={`auth-submit-btn ${isLoading ? 'loading' : ''}`}
                    disabled={isLoading || digits.join('').length < OTP_LENGTH}
                >
                    {isLoading ? 'Verifying…' : 'Verify Email'}
                </button>
            </form>

            {/* Resend */}
            <div className="otp-resend-wrap">
                <span className="otp-resend-label">Didn't receive the code?</span>
                <button
                    type="button"
                    className={`otp-resend-btn ${resendCooldown > 0 ? 'otp-resend-btn--disabled' : ''}`}
                    onClick={handleResend}
                    disabled={resendCooldown > 0 || resendLoading}
                >
                    {resendLoading
                        ? 'Sending…'
                        : resendCooldown > 0
                            ? `Resend in ${resendCooldown}s`
                            : 'Resend Code'}
                </button>
            </div>

            <button type="button" className="otp-back-btn" onClick={onBack}>
                ← Use a different email
            </button>
        </div>
    );
};

// ─── Main AuthModal ───────────────────────────────────────────────────────────
const AuthModal = ({ isOpen, onClose, initialView = 'login' }) => {
    const [view, setView] = useState(initialView);
    const [animKey, setAnimKey] = useState(0);
    const [slideDir, setSlideDir] = useState('right');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [username, setUsername] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    // OTP state
    const [pendingEmail, setPendingEmail] = useState('');

    const { login, register, finalizeLogin } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        setView(initialView);
        setAnimKey(k => k + 1);
        setSlideDir('right');
        setShowPassword(false);
        setShowConfirmPassword(false);
    }, [initialView, isOpen]);

    if (!isOpen) return null;

    const resetFields = () => {
        setEmail('');
        setPassword('');
        setConfirmPassword('');
        setUsername('');
        setShowPassword(false);
        setShowConfirmPassword(false);
        setError('');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        // Manual required-field validation
        if (view === 'register' && !username.trim()) {
            setError('Username is required.');
            return;
        }
        if (!email.trim()) {
            setError('Email address is required.');
            return;
        }
        if (!password) {
            setError('Password is required.');
            return;
        }
        if (view === 'register' && !confirmPassword) {
            setError('Please confirm your password.');
            return;
        }
        if (view === 'register' && password !== confirmPassword) {
            setError('Passwords do not match.');
            return;
        }

        setIsLoading(true);
        try {
            if (view === 'login') {
                await login(email, password);
                onClose();
                navigate('/dashboard');
            } else {
                // register() returns { requires_verification, email }
                const result = await register(username, email, password);
                if (result.requires_verification) {
                    setPendingEmail(result.email);
                    resetFields();
                    // Slide to verify view
                    setSlideDir('left');
                    setAnimKey(k => k + 1);
                    setView('verify');
                }
            }
        } catch (err) {
            setError(err.message || 'An error occurred during authentication');
        } finally {
            setIsLoading(false);
        }
    };

    const handleOTPSuccess = (data) => {
        // data.token + data.user come from the verify-otp endpoint
        finalizeLogin(data.user, data.token);
        onClose();
        navigate('/dashboard');
    };

    const handleOTPBack = () => {
        setPendingEmail('');
        setSlideDir('right');
        setAnimKey(k => k + 1);
        setView('register');
        resetFields();
    };

    const toggleView = () => {
        const goingToRegister = view === 'login';
        setSlideDir(goingToRegister ? 'left' : 'right');
        setAnimKey(k => k + 1);
        setView(goingToRegister ? 'register' : 'login');
        resetFields();
    };

    const handleBackdropClick = (e) => {
        if (e.target.className === 'auth-modal-overlay') onClose();
    };

    const EyeIcon = showPassword ? authIcons['FaEyeSlash'] : authIcons['FaEye'];
    const EyeConfirmIcon = showConfirmPassword ? authIcons['FaEyeSlash'] : authIcons['FaEye'];

    return (
        <div className="auth-modal-overlay" onClick={handleBackdropClick}>
            <div className="auth-modal-content">
                <button className="close-button" onClick={onClose} aria-label="Close">
                    {React.createElement(authIcons['FaTimes'], { size: 24 })}
                </button>

                <div className="auth-header">
                    <Logo variant="modern" className="auth-logo" />
                </div>

                {/* ── OTP Verification View ── */}
                {view === 'verify' ? (
                    <div key={animKey} className={`auth-view-transition auth-slide-${slideDir}`}>
                        <OTPView
                            email={pendingEmail}
                            onSuccess={handleOTPSuccess}
                            onBack={handleOTPBack}
                        />
                    </div>
                ) : (
                    /* ── Login / Register form ── */
                    <div key={animKey} className={`auth-view-transition auth-slide-${slideDir}`}>
                        <div className="auth-view-title">
                            <h2>{view === 'login' ? 'Welcome Back' : 'Create an Account'}</h2>
                            <p>
                                {view === 'login'
                                    ? 'Enter your credentials to access your account'
                                    : 'Sign up to start your career journey'}
                            </p>
                        </div>

                        {error && <div className="auth-error">{error}</div>}

                        <form onSubmit={handleSubmit} className="auth-form" noValidate>
                            {/* Username — register only */}
                            {view === 'register' && (
                                <div className="form-group">
                                    <label htmlFor="username">Username</label>
                                    <div className="input-with-icon">
                                        {React.createElement(authIcons['FaUser'], { className: 'input-icon' })}
                                        <input
                                            type="text"
                                            id="username"
                                            value={username}
                                            onChange={(e) => setUsername(e.target.value)}
                                            placeholder="e.g. johndoe"
                                            required
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Email */}
                            <div className="form-group">
                                <label htmlFor="email">Email Address</label>
                                <div className="input-with-icon">
                                    {React.createElement(authIcons['FaEnvelope'], { className: 'input-icon' })}
                                    <input
                                        type="email"
                                        id="email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        placeholder="you@example.com"
                                        required
                                    />
                                </div>
                            </div>

                            {/* Password */}
                            <div className="form-group">
                                <label htmlFor="password">Password</label>
                                <div className="input-with-icon">
                                    {React.createElement(authIcons['FaLock'], { className: 'input-icon' })}
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        id="password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        placeholder="••••••••"
                                        required
                                    />
                                    <button
                                        type="button"
                                        className="eye-toggle"
                                        onClick={() => setShowPassword(v => !v)}
                                        aria-label={showPassword ? 'Hide password' : 'Show password'}
                                    >
                                        {React.createElement(EyeIcon, { size: 16 })}
                                    </button>
                                </div>
                            </div>

                            {/* Confirm Password — register only */}
                            {view === 'register' && (
                                <div className="form-group">
                                    <label htmlFor="confirmPassword">Confirm Password</label>
                                    <div className="input-with-icon">
                                        {React.createElement(authIcons['FaLock'], { className: 'input-icon' })}
                                        <input
                                            type={showConfirmPassword ? 'text' : 'password'}
                                            id="confirmPassword"
                                            value={confirmPassword}
                                            onChange={(e) => setConfirmPassword(e.target.value)}
                                            placeholder="••••••••"
                                            required
                                        />
                                        <button
                                            type="button"
                                            className="eye-toggle"
                                            onClick={() => setShowConfirmPassword(v => !v)}
                                            aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
                                        >
                                            {React.createElement(EyeConfirmIcon, { size: 16 })}
                                        </button>
                                    </div>
                                </div>
                            )}

                            <button
                                type="submit"
                                className={`auth-submit-btn ${isLoading ? 'loading' : ''}`}
                                disabled={isLoading}
                            >
                                {isLoading
                                    ? 'Processing...'
                                    : view === 'login'
                                        ? 'Sign In'
                                        : 'Create Account'}
                            </button>
                        </form>

                        <div className="auth-footer">
                            <p>
                                {view === 'login' ? "Don't have an account? " : 'Already have an account? '}
                                <button className="toggle-view-btn" onClick={toggleView}>
                                    {view === 'login' ? 'Sign up' : 'Log in'}
                                </button>
                            </p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AuthModal;
