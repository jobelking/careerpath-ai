import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

/**
 * Wraps a route so it's only accessible to authenticated users.
 * Unauthenticated visitors are redirected to "/" (landing page).
 */
const ProtectedRoute = ({ children }) => {
    const { currentUser, loading } = useAuth();

    // While the auth state is being rehydrated from localStorage, render nothing
    if (loading) return null;

    return currentUser ? children : <Navigate to="/" replace />;
};

export default ProtectedRoute;
