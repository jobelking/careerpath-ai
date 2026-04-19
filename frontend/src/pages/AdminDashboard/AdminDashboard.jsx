/**
 * AdminDashboard.jsx
 * ─────────────────
 * Full-featured admin panel for CareerPath AI.
 *
 * Tabs:
 *   • Overview  – high-level stat cards (users, predictions, etc.)
 *   • Users     – searchable/filterable table + Create / Edit / Delete modals
 *   • History   – searchable table of all prediction records + Delete
 *
 * Security:
 *   - This component is mounted behind <AdminRoute> which already checks
 *     currentUser.is_admin before rendering.
 *   - Every API call sends the JWT in the Authorization header.
 *   - All mutation modals require explicit confirmation.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import apiService from '../../services/api/apiService';
import Logo from '../../components/common/Logo';
import { exportToPdf } from '../../utils/exportToPdf';
import { calculateProfileFit, normalizeTop3Fits } from '../../utils/profileFit';
import {
  FiAlertTriangle,
  FiCheckCircle,
  FiDownload,
  FiClock,
  FiEdit2,
  FiFileText,
  FiGrid,
  FiHome,
  FiLogOut,
  FiMenu,
  FiRefreshCw,
  FiShield,
  FiTrash2,
  FiUsers,
} from 'react-icons/fi';
import './AdminDashboard.css';

// ─── Small helper components ───────────────────────────────────────────────────

/** Generic modal wrapper */
const Modal = ({ title, onClose, children, className = '' }) => (
  <div className="adm-modal-overlay" onClick={onClose}>
    <div className={`adm-modal ${className}`} onClick={e => e.stopPropagation()}>
      <div className="adm-modal-header">
        <h3 className="adm-modal-title">{title}</h3>
        <button className="adm-modal-close" onClick={onClose} aria-label="Close">✕</button>
      </div>
      <div className="adm-modal-body">{children}</div>
    </div>
  </div>
);

/** Reusable form field row */
const Field = ({ label, children, error }) => (
  <div className="adm-field">
    <label className="adm-label">{label}</label>
    {children}
    {error && <span className="adm-field-error">{error}</span>}
  </div>
);

/** Stat card used on the Overview tab */
const StatCard = ({ icon, value, label, accent }) => (
  <div className={`adm-stat-card adm-stat-${accent}`}>
    <span className="adm-stat-icon">{icon}</span>
    <span className="adm-stat-value">{value}</span>
    <span className="adm-stat-label">{label}</span>
  </div>
);

/** Inline alert */
const Alert = ({ type, message, onDismiss }) =>
  message ? (
    <div className={`adm-alert adm-alert-${type}`}>
      <span>{message}</span>
      {onDismiss && <button onClick={onDismiss}>✕</button>}
    </div>
  ) : null;

/** Pagination row */
const Pagination = ({ page, total, pageSize, onChange }) => {
  const totalPages = Math.ceil(total / pageSize) || 1;
  return (
    <div className="adm-pagination">
      <span className="adm-page-info">
        {Math.min((page - 1) * pageSize + 1, total)}–{Math.min(page * pageSize, total)} of {total}
      </span>
      <button disabled={page <= 1} onClick={() => onChange(page - 1)}>‹ Prev</button>
      <span className="adm-page-current">Page {page} / {totalPages}</span>
      <button disabled={page >= totalPages} onClick={() => onChange(page + 1)}>Next ›</button>
    </div>
  );
};

// ─── Main component ────────────────────────────────────────────────────────────

const AdminDashboard = () => {
  const navigate = useNavigate();
  const { currentUser, logout, getToken } = useAuth();

  // Active navigation tab: 'overview' | 'users' | 'history'
  const [activeTab, setActiveTab] = useState('overview');
  // Mobile sidebar toggle
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // ── Global alert state ──────────────────────────────────────────────────────
  const [alert, setAlert] = useState({ type: '', message: '' });

  const showAlert = (type, message) => {
    setAlert({ type, message });
    // Auto-dismiss success alerts after 4 s
    if (type === 'success') setTimeout(() => setAlert({ type: '', message: '' }), 4000);
  };

  // ── Token helper ────────────────────────────────────────────────────────────
  const token = getToken();

  // ════════════════════════════════════════════════════════════════════════════
  // OVERVIEW / STATS
  // ════════════════════════════════════════════════════════════════════════════

  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);

  const loadStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      const data = await apiService.getAdminStats(token);
      setStats(data);
    } catch (err) {
      showAlert('error', `Failed to load stats: ${err.message}`);
    } finally {
      setStatsLoading(false);
    }
  }, [token]);

  // ════════════════════════════════════════════════════════════════════════════
  // USERS TAB STATE
  // ════════════════════════════════════════════════════════════════════════════

  const [users, setUsers] = useState([]);
  const [userTotal, setUserTotal] = useState(0);
  const [userPage, setUserPage] = useState(1);
  const PAGE_SIZE = 15;

  // Filter/search state
  const [userSearch, setUserSearch] = useState('');
  const [userVerified, setUserVerified] = useState('all');  // all | true | false
  const [userAdminFilter, setUserAdminFilter] = useState('all'); // all | true | false
  const [usersLoading, setUsersLoading] = useState(false);

  // Modals
  const [showCreateUser, setShowCreateUser] = useState(false);
  const [editingUser, setEditingUser] = useState(null);   // user object | null
  const [deletingUser, setDeletingUser] = useState(null);   // user object | null

  // Create/edit form fields
  const emptyUserForm = { username: '', email: '', password: '', is_admin: false, is_verified: true };
  const [userForm, setUserForm] = useState(emptyUserForm);
  const [formErrors, setFormErrors] = useState({});
  const [formLoading, setFormLoading] = useState(false);

  /** Fetch users from the backend */
  const loadUsers = useCallback(async (page = userPage) => {
    setUsersLoading(true);
    try {
      const data = await apiService.getAdminUsers(token, {
        search: userSearch,
        verified: userVerified,
        is_admin: userAdminFilter,
        page,
        page_size: PAGE_SIZE,
      });
      setUsers(data.users);
      setUserTotal(data.total);
    } catch (err) {
      showAlert('error', `Failed to load users: ${err.message}`);
    } finally {
      setUsersLoading(false);
    }
  }, [token, userSearch, userVerified, userAdminFilter, userPage]);

  /** Validate create/edit user form fields */
  const validateUserForm = (isCreate) => {
    const errs = {};
    if (!userForm.username.trim()) errs.username = 'Username is required.';
    if (!userForm.email.trim()) errs.email = 'Email is required.';
    else if (!/\S+@\S+\.\S+/.test(userForm.email)) errs.email = 'Invalid email address.';
    if (isCreate && !userForm.password) errs.password = 'Password is required.';
    if (userForm.password && userForm.password.length < 6) errs.password = 'Minimum 6 characters.';
    return errs;
  };

  /** Submit the Create User form */
  const handleCreateUser = async (e) => {
    e.preventDefault();
    const errs = validateUserForm(true);
    if (Object.keys(errs).length) { setFormErrors(errs); return; }
    setFormLoading(true);
    try {
      await apiService.adminCreateUser(token, {
        username: userForm.username.trim(),
        email: userForm.email.trim(),
        password: userForm.password,
        is_admin: userForm.is_admin,
      });
      showAlert('success', `User "${userForm.username.trim()}" created successfully.`);
      setShowCreateUser(false);
      setUserForm(emptyUserForm);
      setFormErrors({});
      loadUsers(1);
      loadStats();
    } catch (err) {
      showAlert('error', err.message);
    } finally {
      setFormLoading(false);
    }
  };

  /** Submit the Edit User form */
  const handleUpdateUser = async (e) => {
    e.preventDefault();
    const errs = validateUserForm(false);
    if (Object.keys(errs).length) { setFormErrors(errs); return; }
    setFormLoading(true);
    try {
      // Only send changed fields
      const payload = {};
      if (userForm.username !== editingUser.username) payload.username = userForm.username.trim();
      if (userForm.email !== editingUser.email) payload.email = userForm.email.trim();
      if (userForm.is_admin !== editingUser.is_admin) payload.is_admin = userForm.is_admin;
      if (userForm.is_verified !== editingUser.is_verified) payload.is_verified = userForm.is_verified;
      if (userForm.password) payload.password = userForm.password;

      if (!Object.keys(payload).length) {
        showAlert('info', 'No changes detected.');
        setEditingUser(null);
        setFormLoading(false);
        return;
      }

      await apiService.adminUpdateUser(token, editingUser.id, payload);
      showAlert('success', `User "${editingUser.username}" updated.`);
      setEditingUser(null);
      setUserForm(emptyUserForm);
      setFormErrors({});
      loadUsers(userPage);
    } catch (err) {
      showAlert('error', err.message);
    } finally {
      setFormLoading(false);
    }
  };

  /** Execute user deletion after confirmation */
  const handleDeleteUser = async () => {
    if (!deletingUser) return;
    setFormLoading(true);
    try {
      await apiService.adminDeleteUser(token, deletingUser.id);
      showAlert('success', `User "${deletingUser.username}" deleted.`);
      setDeletingUser(null);
      // Go back a page if we deleted the last item on the current page
      const newPage = users.length === 1 && userPage > 1 ? userPage - 1 : userPage;
      setUserPage(newPage);
      loadUsers(newPage);
      loadStats();
    } catch (err) {
      showAlert('error', err.message);
    } finally {
      setFormLoading(false);
    }
  };

  /** Open edit modal pre-populated with user data */
  const openEditUser = (user) => {
    setEditingUser(user);
    setUserForm({
      username: user.username,
      email: user.email,
      password: '',               // never pre-fill password
      is_admin: user.is_admin,
      is_verified: user.is_verified,
    });
    setFormErrors({});
  };

  // ════════════════════════════════════════════════════════════════════════════
  // HISTORY TAB STATE
  // ════════════════════════════════════════════════════════════════════════════

  const [history, setHistory] = useState([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyPage, setHistoryPage] = useState(1);
  const [historySearch, setHistorySearch] = useState('');
  const [historyUserId, setHistoryUserId] = useState(0);
  const [historyUsers, setHistoryUsers] = useState([]);
  const [historyUsersLoading, setHistoryUsersLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [exportingRecordId, setExportingRecordId] = useState(null);
  const [deletingRecord, setDeletingRecord] = useState(null); // history record | null
  const [deleteRecordLoading, setDeleteRecordLoading] = useState(false);
  const [loadingResumeId, setLoadingResumeId] = useState(null); // record id being fetched
  const [resumeModal, setResumeModal] = useState(null); // { rec, blobUrl } | null

  const handleCloseResumeModal = () => {
    if (resumeModal?.blobUrl) URL.revokeObjectURL(resumeModal.blobUrl);
    setResumeModal(null);
  };

  /** Fetch history records from the backend */
  const loadHistory = useCallback(async (page = historyPage) => {
    setHistoryLoading(true);
    try {
      const data = await apiService.getAdminHistory(token, {
        search: historySearch,
        user_id: historyUserId,
        page,
        page_size: PAGE_SIZE,
      });
      setHistory(data.history);
      setHistoryTotal(data.total);
    } catch (err) {
      showAlert('error', `Failed to load history: ${err.message}`);
    } finally {
      setHistoryLoading(false);
    }
  }, [token, historySearch, historyUserId, historyPage]);

  /** Fetch users who have prediction history (for filter dropdown) */
  const loadHistoryUsers = useCallback(async () => {
    setHistoryUsersLoading(true);
    try {
      const data = await apiService.getAdminHistoryUsers(token);
      setHistoryUsers(data.users || []);
    } catch (err) {
      showAlert('error', `Failed to load history users: ${err.message}`);
    } finally {
      setHistoryUsersLoading(false);
    }
  }, [token]);

  /** Execute history record deletion */
  const handleDeleteRecord = async () => {
    if (!deletingRecord) return;
    setDeleteRecordLoading(true);
    try {
      await apiService.adminDeleteHistory(token, deletingRecord.id);
      showAlert('success', `History record #${deletingRecord.id} deleted.`);
      setDeletingRecord(null);
      const newPage = history.length === 1 && historyPage > 1 ? historyPage - 1 : historyPage;
      setHistoryPage(newPage);
      loadHistory(newPage);
      loadStats();
    } catch (err) {
      showAlert('error', err.message);
    } finally {
      setDeleteRecordLoading(false);
    }
  };

  const handleExportRecordPdf = async (record) => {
    setExportingRecordId(record.id);
    try {
      const topThree = Array.isArray(record.top_predictions) && record.top_predictions.length
        ? record.top_predictions.slice(0, 3)
        : [{
          career_path: record.prediction_result,
          raw_confidence: record.confidence_score ?? 0,
        }];

      const selectedCareer = topThree[0]?.career_path || record.prediction_result;
      const roadmapByPath = record.learning_roadmap_by_path ?? null;
      const certsByPath = record.certification_data_by_path ?? null;
      const keywordsByPath = record.extracted_keywords_by_path ?? null;

      await exportToPdf({
        topThree,
        calculateProfileFit,
        learningRoadmap: roadmapByPath?.[selectedCareer] ?? record.learning_roadmap ?? null,
        certificationData: certsByPath?.[selectedCareer] ?? record.certification_data ?? null,
        careerContent: null,
        logoRef: null,
        extractedKeywords: keywordsByPath?.[selectedCareer] ?? (Array.isArray(record.extracted_keywords) ? record.extracted_keywords : []),
        selectedCareerPath: selectedCareer,
      });
    } catch (err) {
      showAlert('error', `Failed to export report: ${err.message}`);
    } finally {
      setExportingRecordId(null);
    }
  };

  // ════════════════════════════════════════════════════════════════════════════
  // DATA LOADING EFFECTS
  // ════════════════════════════════════════════════════════════════════════════

  // Load stats whenever Overview tab is active
  useEffect(() => {
    if (activeTab === 'overview') loadStats();
  }, [activeTab, loadStats]);

  // Load users whenever filters, search, or page changes on Users tab
  useEffect(() => {
    if (activeTab === 'users') loadUsers(userPage);
  }, [activeTab, userSearch, userVerified, userAdminFilter, userPage, loadUsers]);

  // Load history whenever search or page changes on History tab
  useEffect(() => {
    if (activeTab === 'history') loadHistory(historyPage);
  }, [activeTab, historySearch, historyUserId, historyPage, loadHistory]);

  // Load list of users with predictions for History filter
  useEffect(() => {
    if (activeTab === 'history' && historyUsers.length === 0 && !historyUsersLoading) {
      loadHistoryUsers();
    }
  }, [activeTab, historyUsers.length, historyUsersLoading, loadHistoryUsers]);

  // Debounce user search input (300 ms) to avoid hammering the API
  useEffect(() => {
    const timer = setTimeout(() => {
      if (activeTab === 'users') { setUserPage(1); loadUsers(1); }
    }, 300);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userSearch]);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (activeTab === 'history') { setHistoryPage(1); loadHistory(1); }
    }, 300);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [historySearch]);

  const getTopPredictions = (record) => {
    if (Array.isArray(record?.top_predictions) && record.top_predictions.length) {
      return record.top_predictions.slice(0, 3);
    }
    return record?.prediction_result
      ? [{ career_path: record.prediction_result, raw_confidence: record.confidence_score }]
      : [];
  };

  const getHistoryUserLabel = (user) => {
    if (!user) return 'All Users';
    const name = user.username || 'Unknown';
    const email = user.email ? ` (${user.email})` : '';
    return `${name}${email}`;
  };

  // ── Logout ──────────────────────────────────────────────────────────────────
  const handleLogout = () => {
    logout();
    navigate('/');
  };

  // ════════════════════════════════════════════════════════════════════════════
  // FORMAT HELPERS
  // ════════════════════════════════════════════════════════════════════════════

  const fmtDate = (iso) => {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
    });
  };

  const fmtConfidence = (score) =>
    score != null ? `${parseFloat(score).toFixed(1)}%` : '—';

  // calculateProfileFit is now imported from '../../utils/profileFit'


  // ════════════════════════════════════════════════════════════════════════════
  // RENDER
  // ════════════════════════════════════════════════════════════════════════════

  return (
    <div className="adm-layout">

      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <aside className={`adm-sidebar ${sidebarOpen ? 'adm-sidebar-open' : ''}`}>
        {/* Brand */}
        <div className="adm-sidebar-brand" onClick={() => navigate('/dashboard')}>
          <Logo variant="modern" className="adm-brand-logo" />
        </div>

        {/* Nav items */}
        <nav className="adm-nav">
          {[
            { id: 'overview', icon: <FiGrid />, label: 'Overview' },
            { id: 'users', icon: <FiUsers />, label: 'Users' },
            { id: 'history', icon: <FiFileText />, label: 'History' },
          ].map(({ id, icon, label }) => (
            <button
              key={id}
              className={`adm-nav-item ${activeTab === id ? 'adm-nav-active' : ''}`}
              onClick={() => { setActiveTab(id); setSidebarOpen(false); }}
            >
              <span className="adm-nav-icon">{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </nav>

        {/* Sidebar footer */}
        <div className="adm-sidebar-footer">
          <button className="adm-nav-item adm-nav-back" onClick={() => navigate('/dashboard')}>
            <span className="adm-nav-icon"><FiHome /></span>
            <span>App Dashboard</span>
          </button>
          <button className="adm-nav-item adm-nav-logout" onClick={handleLogout}>
            <span className="adm-nav-icon"><FiLogOut /></span>
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* Sidebar overlay (mobile) */}
      {sidebarOpen && (
        <div className="adm-sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* ── Main content ────────────────────────────────────────────────────── */}
      <main className="adm-main">

        {/* Top bar */}
        <header className="adm-topbar">
          <div className="adm-topbar-left">
            <button
              className="adm-hamburger"
              onClick={() => setSidebarOpen(s => !s)}
              aria-label="Toggle sidebar"
            >
              <FiMenu />
            </button>
            <Logo variant="modern" className="adm-topbar-logo" />
            <div className="adm-breadcrumb">
              <span className="adm-breadcrumb-root">Admin</span>
              <span className="adm-breadcrumb-sep">›</span>
              <span className="adm-breadcrumb-page">
                {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
              </span>
            </div>
          </div>
          <div className="adm-topbar-right">
            <span className="adm-admin-badge">Admin</span>
            <span className="adm-admin-name">{currentUser?.username}</span>
          </div>
        </header>

        {/* Global alert */}
        <div className="adm-alert-wrapper">
          <Alert
            type={alert.type}
            message={alert.message}
            onDismiss={() => setAlert({ type: '', message: '' })}
          />
        </div>

        {/* ════════════════════════════════════════════════════════ */}
        {/* OVERVIEW TAB                                            */}
        {/* ════════════════════════════════════════════════════════ */}
        {activeTab === 'overview' && (
          <section className="adm-section">
            <h2 className="adm-section-title">Dashboard Overview</h2>

            {statsLoading ? (
              <div className="adm-loading">Loading statistics…</div>
            ) : stats ? (
              <>
                <div className="adm-stats-grid">
                  <StatCard icon={<FiUsers />} value={stats.total_users} label="Total Users" accent="blue" />
                  <StatCard icon={<FiCheckCircle />} value={stats.verified_users} label="Verified Users" accent="green" />
                  <StatCard icon={<FiShield />} value={stats.admin_users} label="Admin Accounts" accent="purple" />
                  <StatCard icon={<FiFileText />} value={stats.total_predictions} label="Total Predictions" accent="orange" />
                </div>

                {/* Quick-action buttons */}
                <div className="adm-quick-actions">
                  <h3 className="adm-quick-title">Quick Actions</h3>
                  <div className="adm-quick-btns">
                    <button className="adm-btn adm-btn-primary" onClick={() => { setActiveTab('users'); setShowCreateUser(true); }}>
                      + Add New User
                    </button>
                    <button className="adm-btn adm-btn-secondary" onClick={() => setActiveTab('users')}>
                      Manage Users
                    </button>
                    <button className="adm-btn adm-btn-secondary" onClick={() => setActiveTab('history')}>
                      View All History
                    </button>
                    <button className="adm-btn adm-btn-ghost" onClick={loadStats}>
                      <FiRefreshCw /> Refresh Stats
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="adm-empty">No statistics available.</div>
            )}
          </section>
        )}

        {/* ════════════════════════════════════════════════════════ */}
        {/* USERS TAB                                               */}
        {/* ════════════════════════════════════════════════════════ */}
        {activeTab === 'users' && (
          <section className="adm-section">
            {/* Section header */}
            <div className="adm-section-header">
              <h2 className="adm-section-title">User Management</h2>
              <button className="adm-btn adm-btn-primary" onClick={() => { setUserForm(emptyUserForm); setFormErrors({}); setShowCreateUser(true); }}>
                + New User
              </button>
            </div>

            {/* Filters row */}
            <div className="adm-filters">
              <input
                className="adm-search"
                type="text"
                placeholder="Search by name or email…"
                value={userSearch}
                onChange={e => setUserSearch(e.target.value)}
              />
              <select
                className="adm-select"
                value={userVerified}
                onChange={e => { setUserVerified(e.target.value); setUserPage(1); }}
              >
                <option value="all">All Verified Status</option>
                <option value="true">Verified</option>
                <option value="false">Unverified</option>
              </select>
              <select
                className="adm-select"
                value={userAdminFilter}
                onChange={e => { setUserAdminFilter(e.target.value); setUserPage(1); }}
              >
                <option value="all">All Roles</option>
                <option value="true">Admins Only</option>
                <option value="false">Regular Users</option>
              </select>
              <button className="adm-btn adm-btn-ghost" onClick={() => loadUsers(userPage)}>
                <FiRefreshCw /> Refresh
              </button>
            </div>

            {/* Table */}
            {usersLoading ? (
              <div className="adm-loading">Loading users…</div>
            ) : users.length === 0 ? (
              <div className="adm-empty">No users found.</div>
            ) : (
              <>
                <div className="adm-table-wrapper">
                  <table className="adm-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Username</th>
                        <th>Email</th>
                        <th>Verified</th>
                        <th>Role</th>
                        <th>Created</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {users.map(user => (
                        <tr key={user.id} className={user.id === currentUser?.id ? 'adm-row-self' : ''}>
                          <td className="adm-cell-muted">#{user.id}</td>
                          <td><strong>{user.username}</strong></td>
                          <td>{user.email}</td>
                          <td>
                            <span className={`adm-badge ${user.is_verified ? 'adm-badge-green' : 'adm-badge-yellow'}`}>
                              {user.is_verified ? (
                                <span className="adm-badge-content"><FiCheckCircle /> Verified</span>
                              ) : (
                                <span className="adm-badge-content"><FiClock /> Pending</span>
                              )}
                            </span>
                          </td>
                          <td>
                            <span className={`adm-badge ${user.is_admin ? 'adm-badge-purple' : 'adm-badge-gray'}`}>
                              {user.is_admin ? (
                                <span className="adm-badge-content"><FiShield /> Admin</span>
                              ) : 'User'}
                            </span>
                          </td>
                          <td className="adm-cell-muted">{fmtDate(user.created_at)}</td>
                          <td>
                            <div className="adm-actions">
                              <button className="adm-btn-icon adm-btn-edit" onClick={() => openEditUser(user)} title="Edit user">
                                <FiEdit2 />
                              </button>
                              <button
                                className="adm-btn-icon adm-btn-delete"
                                onClick={() => setDeletingUser(user)}
                                title="Delete user"
                                disabled={user.id === currentUser?.id}
                              >
                                <FiTrash2 />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <Pagination
                  page={userPage}
                  total={userTotal}
                  pageSize={PAGE_SIZE}
                  onChange={p => setUserPage(p)}
                />
              </>
            )}
          </section>
        )}

        {/* ════════════════════════════════════════════════════════ */}
        {/* HISTORY TAB                                             */}
        {/* ════════════════════════════════════════════════════════ */}
        {activeTab === 'history' && (
          <section className="adm-section">
            <div className="adm-section-header">
              <h2 className="adm-section-title">Prediction History</h2>
            </div>

            {/* Filter row */}
            <div className="adm-filters">
              <input
                className="adm-search"
                type="text"
                placeholder="Search by career path or filename…"
                value={historySearch}
                onChange={e => setHistorySearch(e.target.value)}
              />
              <select
                className="adm-select"
                value={historyUserId}
                onChange={e => { setHistoryUserId(Number(e.target.value)); setHistoryPage(1); }}
              >
                <option value={0}>All Users</option>
                {historyUsers.map(user => (
                  <option key={user.id} value={user.id}>
                    {getHistoryUserLabel(user)} ({user.total_predictions})
                  </option>
                ))}
              </select>
              <button className="adm-btn adm-btn-ghost" onClick={() => loadHistory(historyPage)}>
                <FiRefreshCw /> Refresh
              </button>
            </div>

            {historyUserId !== 0 && (
              <div className="adm-filter-pill">
                <span>Filtered by:</span>
                <strong>
                  {getHistoryUserLabel(historyUsers.find(u => u.id === historyUserId))}
                </strong>
                <button
                  type="button"
                  className="adm-pill-clear"
                  onClick={() => { setHistoryUserId(0); setHistoryPage(1); }}
                >
                  Clear
                </button>
              </div>
            )}

            {historyLoading ? (
              <div className="adm-loading">Loading history…</div>
            ) : history.length === 0 ? (
              <div className="adm-empty">No prediction records found.</div>
            ) : (
              <>
                <div className="adm-table-wrapper">
                  <table className="adm-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>User</th>
                        <th>Predicted Career</th>
                        <th>Top 3 Predictions</th>
                        <th>Profile Fit</th>
                        <th>File</th>
                        <th>Date</th>
                        <th>Resume</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.map(rec => (
                        <tr key={rec.id}>
                          <td className="adm-cell-muted">#{rec.id}</td>
                          <td>
                            <div className="adm-cell-user">
                              <span className="adm-cell-username">{rec.username || '—'}</span>
                              <span className="adm-cell-email">{rec.user_email}</span>
                            </div>
                          </td>
                          <td>
                            <span className="adm-career-tag">{rec.prediction_result}</span>
                          </td>
                          <td>
                            <div className="adm-top-preds">
                              {(() => {
                                const topPreds = getTopPredictions(rec);
                                if (topPreds.length === 0) {
                                  return <span className="adm-cell-muted">—</span>;
                                }
                                return topPreds.map((pred, idx) => (
                                  <div key={`${rec.id}-${idx}`} className="adm-top-pred-row">
                                    <span className="adm-top-pred-rank">#{idx + 1}</span>
                                    <span className="adm-top-pred-name">{pred.career_path || '—'}</span>
                                    {pred.raw_confidence != null && (
                                      <span className="adm-top-pred-score">{fmtConfidence(pred.raw_confidence)}</span>
                                    )}
                                  </div>
                                ));
                              })()}
                            </div>
                          </td>
                          <td>
                            {(() => {
                              if (rec.confidence_score == null) return '—';
                              const topPreds = getTopPredictions(rec);
                              const nFits = normalizeTop3Fits(topPreds);
                              const topFit = nFits[0] ?? calculateProfileFit(parseFloat(rec.confidence_score));
                              return `${topFit}%`;
                            })()}
                          </td>
                          <td className="adm-cell-muted">{rec.filename || '—'}</td>
                          <td className="adm-cell-muted">{fmtDate(rec.date_created)}</td>
                          <td>
                            {rec.has_resume ? (
                              <button
                                className="adm-btn adm-btn-ghost adm-btn-sm"
                                disabled={loadingResumeId === rec.id}
                                onClick={async () => {
                                  setLoadingResumeId(rec.id);
                                  try {
                                    const response = await fetch(
                                      `${(import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')}/api/admin/history/${rec.id}/resume`,
                                      { headers: { Authorization: `Bearer ${token}` } }
                                    );
                                    const data = await response.json();
                                    if (!response.ok) throw new Error(data.detail || 'Failed to load resume');
                                    // Open the signed Supabase URL in a new tab
                                    window.open(data.url, '_blank');
                                  } catch (err) {
                                    showAlert('error', `Could not load resume: ${err.message}`);
                                  } finally {
                                    setLoadingResumeId(null);
                                  }
                                }}
                              >
                                {loadingResumeId === rec.id ? <FiClock /> : 'View'}
                              </button>
                            ) : (
                              <span className="adm-cell-muted">—</span>
                            )}
                          </td>
                          <td>
                            <button
                              className="adm-btn-icon adm-btn-export"
                              title="Export report PDF"
                              onClick={() => handleExportRecordPdf(rec)}
                              disabled={exportingRecordId === rec.id}
                            >
                              {exportingRecordId === rec.id ? <FiClock /> : <FiDownload />}
                            </button>
                            <button
                              className="adm-btn-icon adm-btn-delete"
                              onClick={() => setDeletingRecord(rec)}
                              title="Delete record"
                            >
                              <FiTrash2 />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <Pagination
                  page={historyPage}
                  total={historyTotal}
                  pageSize={PAGE_SIZE}
                  onChange={p => setHistoryPage(p)}
                />
              </>
            )}
          </section>
        )}
      </main>

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* MODALS                                                              */}
      {/* ════════════════════════════════════════════════════════════════════ */}

      {/* ── Create User Modal ─────────────────────────────────────────────── */}
      {showCreateUser && (
        <Modal title="Create New User" onClose={() => setShowCreateUser(false)}>
          <form onSubmit={handleCreateUser} noValidate>
            <Field label="Username *" error={formErrors.username}>
              <input
                className="adm-input"
                value={userForm.username}
                onChange={e => setUserForm(f => ({ ...f, username: e.target.value }))}
                placeholder="e.g. john_doe"
                autoFocus
              />
            </Field>
            <Field label="Email *" error={formErrors.email}>
              <input
                className="adm-input"
                type="email"
                value={userForm.email}
                onChange={e => setUserForm(f => ({ ...f, email: e.target.value }))}
                placeholder="john@example.com"
              />
            </Field>
            <Field label="Password *" error={formErrors.password}>
              <input
                className="adm-input"
                type="password"
                value={userForm.password}
                onChange={e => setUserForm(f => ({ ...f, password: e.target.value }))}
                placeholder="Min 6 characters"
              />
            </Field>
            <div className="adm-checkbox-row">
              <label className="adm-checkbox-label">
                <input
                  type="checkbox"
                  checked={userForm.is_admin}
                  onChange={e => setUserForm(f => ({ ...f, is_admin: e.target.checked }))}
                />
                Grant admin privileges
              </label>
            </div>
            <div className="adm-modal-actions">
              <button type="button" className="adm-btn adm-btn-ghost" onClick={() => setShowCreateUser(false)}>
                Cancel
              </button>
              <button type="submit" className="adm-btn adm-btn-primary" disabled={formLoading}>
                {formLoading ? 'Creating…' : 'Create User'}
              </button>
            </div>
          </form>
        </Modal>
      )}

      {/* ── Edit User Modal ───────────────────────────────────────────────── */}
      {editingUser && (
        <Modal title={`Edit User — ${editingUser.username}`} onClose={() => setEditingUser(null)}>
          <form onSubmit={handleUpdateUser} noValidate>
            <Field label="Username *" error={formErrors.username}>
              <input
                className="adm-input"
                value={userForm.username}
                onChange={e => setUserForm(f => ({ ...f, username: e.target.value }))}
              />
            </Field>
            <Field label="Email *" error={formErrors.email}>
              <input
                className="adm-input"
                type="email"
                value={userForm.email}
                onChange={e => setUserForm(f => ({ ...f, email: e.target.value }))}
              />
            </Field>
            <Field label="New Password" error={formErrors.password}>
              <input
                className="adm-input"
                type="password"
                value={userForm.password}
                onChange={e => setUserForm(f => ({ ...f, password: e.target.value }))}
                placeholder="Leave blank to keep current password"
              />
            </Field>
            <div className="adm-checkbox-row">
              <label className="adm-checkbox-label">
                <input
                  type="checkbox"
                  checked={userForm.is_admin}
                  onChange={e => setUserForm(f => ({ ...f, is_admin: e.target.checked }))}
                  disabled={editingUser.id === currentUser?.id} // can't remove own admin
                />
                Admin privileges
              </label>
              <label className="adm-checkbox-label">
                <input
                  type="checkbox"
                  checked={userForm.is_verified}
                  onChange={e => setUserForm(f => ({ ...f, is_verified: e.target.checked }))}
                />
                Email verified
              </label>
            </div>
            <div className="adm-modal-actions">
              <button type="button" className="adm-btn adm-btn-ghost" onClick={() => setEditingUser(null)}>
                Cancel
              </button>
              <button type="submit" className="adm-btn adm-btn-primary" disabled={formLoading}>
                {formLoading ? 'Saving…' : 'Save Changes'}
              </button>
            </div>
          </form>
        </Modal>
      )}
      {/* ── Resume PDF Viewer ────────────────────────────────────────────────── */}
      {resumeModal && (
        <Modal
          title={`Resume — ${resumeModal.rec.filename || `Record #${resumeModal.rec.id}`}`}
          onClose={handleCloseResumeModal}
          className="adm-modal-pdf"
        >
          <iframe
            className="adm-pdf-frame"
            src={resumeModal.blobUrl}
            title="Resume PDF"
          />
        </Modal>
      )}
      {/* ── Delete User Confirmation ──────────────────────────────────────── */}
      {deletingUser && (
        <Modal title="Confirm Delete" onClose={() => setDeletingUser(null)}>
          <div className="adm-confirm-body">
            <span className="adm-confirm-icon"><FiAlertTriangle /></span>
            <p>
              You are about to permanently delete user{' '}
              <strong>{deletingUser.username}</strong> ({deletingUser.email}).
            </p>
            <p className="adm-confirm-sub">
              All their prediction history will also be removed. This action cannot be undone.
            </p>
          </div>
          <div className="adm-modal-actions">
            <button className="adm-btn adm-btn-ghost" onClick={() => setDeletingUser(null)}>
              Cancel
            </button>
            <button className="adm-btn adm-btn-danger" onClick={handleDeleteUser} disabled={formLoading}>
              {formLoading ? 'Deleting…' : 'Yes, Delete'}
            </button>
          </div>
        </Modal>
      )}

      {/* ── Delete History Record Confirmation ───────────────────────────── */}
      {deletingRecord && (
        <Modal title="Delete History Record" onClose={() => setDeletingRecord(null)}>
          <div className="adm-confirm-body">
            <span className="adm-confirm-icon"><FiAlertTriangle /></span>
            <p>
              Delete prediction record <strong>#{deletingRecord.id}</strong> (
              {deletingRecord.prediction_result})?
            </p>
            <p className="adm-confirm-sub">This action cannot be undone.</p>
          </div>
          <div className="adm-modal-actions">
            <button className="adm-btn adm-btn-ghost" onClick={() => setDeletingRecord(null)}>
              Cancel
            </button>
            <button className="adm-btn adm-btn-danger" onClick={handleDeleteRecord} disabled={deleteRecordLoading}>
              {deleteRecordLoading ? 'Deleting…' : 'Yes, Delete'}
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
};

export default AdminDashboard;
