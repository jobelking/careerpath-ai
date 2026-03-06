import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

/**
 * AdminRoute — wraps a route so it's only accessible to admin users.
 * - Unauthenticated users are redirected to "/" (landing page).
 * - Authenticated non-admin users are redirected to "/dashboard".
 * - While auth is loading, renders nothing (prevents flash).
 */
const AdminRoute = ({ children }) => {
    const { currentUser, loading } = useAuth();

    // Wait for session rehydration before making routing decisions
    if (loading) return null;

    // Not logged in → back to landing
    if (!currentUser) return <Navigate to="/" replace />;

    // Logged in but not an admin → back to user dashboard
    if (!currentUser.is_admin) return <Navigate to="/dashboard" replace />;

    return children;
};

export default AdminRoute;
