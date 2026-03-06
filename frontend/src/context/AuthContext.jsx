import React, { createContext, useState, useContext, useEffect } from 'react';
import apiService from '../services/api/apiService';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

const TOKEN_KEY = 'careerpath_token';
const USER_KEY = 'careerpath_user';

export const AuthProvider = ({ children }) => {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // On mount: rehydrate session from stored JWT
    useEffect(() => {
        const rehydrateSession = async () => {
            const token = localStorage.getItem(TOKEN_KEY);
            if (!token) {
                setLoading(false);
                return;
            }
            try {
                // Verify token is still valid by hitting /api/auth/me
                const user = await apiService.getCurrentUser(token);
                setCurrentUser(user);
            } catch {
                // Token expired or invalid — clear stored data
                localStorage.removeItem(TOKEN_KEY);
                localStorage.removeItem(USER_KEY);
            } finally {
                setLoading(false);
            }
        };
        rehydrateSession();
    }, []);

    /**
     * Register a new user.
     * Returns { requires_verification: true, email } — NO JWT yet.
     * JWT is only issued after OTP is verified.
     */
    const register = async (name, email, password) => {
        const data = await apiService.registerUser(name, email, password);
        // Don't store token/user yet — must verify email first
        return { requires_verification: data.requires_verification, email };
    };

    /**
     * Login with email and password.
     * Calls POST /api/auth/login, stores JWT + user data.
     */
    const login = async (email, password) => {
        const data = await apiService.loginUser(email, password);
        localStorage.setItem(TOKEN_KEY, data.token);
        localStorage.setItem(USER_KEY, JSON.stringify(data.user));
        setCurrentUser(data.user);
        return data.user;
    };

    /**
     * Finalize login after OTP verification.
     * Called when verifyOTP returns a JWT.
     */
    const finalizeLogin = (userData, token) => {
        localStorage.setItem(TOKEN_KEY, token);
        localStorage.setItem(USER_KEY, JSON.stringify(userData));
        setCurrentUser(userData);
    };

    /**
     * Logout — clears JWT and user from state + localStorage.
     */
    const logout = () => {
        setCurrentUser(null);
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        // Clear dashboard session data so resume/results don't persist to the next user
        sessionStorage.removeItem('careerpath_dashboard');
        // Notify DashboardContext (and any other listeners) to wipe in-memory state
        window.dispatchEvent(new CustomEvent('careerpath:logout'));
    };

    /**
     * Get the stored JWT token (useful for authenticated API calls).
     */
    const getToken = () => localStorage.getItem(TOKEN_KEY);

    const value = {
        currentUser,
        login,
        register,
        finalizeLogin,
        logout,
        getToken,
        loading,
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};
