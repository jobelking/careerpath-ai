import React, { createContext, useContext, useState, useEffect } from 'react';

const DashboardContext = createContext();

const SESSION_KEY = 'careerpath_dashboard';

// Read persisted state from sessionStorage once on load
const loadFromSession = () => {
    try {
        const raw = sessionStorage.getItem(SESSION_KEY);
        return raw ? JSON.parse(raw) : {};
    } catch {
        return {};
    }
};

export const useDashboard = () => {
    const context = useContext(DashboardContext);
    if (!context) {
        throw new Error('useDashboard must be used within a DashboardProvider');
    }
    return context;
};

export const DashboardProvider = ({ children }) => {
    const persisted = loadFromSession();

    const [predictionResults, setPredictionResults] = useState(persisted.predictionResults ?? null);
    const [uploadedFileName, setUploadedFileName] = useState(persisted.uploadedFileName ?? null);
    // File objects cannot be serialized; always start null (user can re-pick if needed)
    const [uploadedFile, setUploadedFile] = useState(null);
    const [resumeText, setResumeText] = useState(persisted.resumeText ?? null);
    const [learningRoadmapByPath, setLearningRoadmapByPath] = useState(persisted.learningRoadmapByPath ?? null);
    const [certificationDataByPath, setCertificationDataByPath] = useState(persisted.certificationDataByPath ?? null);
    // ID of the latest prediction_history record — needed to PATCH roadmap/certs later
    const [historyRecordId, setHistoryRecordId] = useState(persisted.historyRecordId ?? null);

    // Sync serializable state to sessionStorage whenever it changes
    useEffect(() => {
        try {
            sessionStorage.setItem(SESSION_KEY, JSON.stringify({
                predictionResults,
                uploadedFileName,
                resumeText,
                learningRoadmapByPath,
                certificationDataByPath,
                historyRecordId,
            }));
        } catch {
            // Ignore storage quota errors
        }
    }, [predictionResults, uploadedFileName, resumeText, learningRoadmapByPath, certificationDataByPath, historyRecordId]);

    // When the user logs out, wipe all in-memory dashboard state immediately.
    // sessionStorage is already cleared by AuthContext.logout(), but React state
    // persists in memory until explicitly reset — causing data leakage to the next user.
    useEffect(() => {
        const handleLogout = () => {
            setPredictionResults(null);
            setUploadedFileName(null);
            setUploadedFile(null);
            setResumeText(null);
            setLearningRoadmapByPath(null);
            setCertificationDataByPath(null);
            setHistoryRecordId(null);
        };
        window.addEventListener('careerpath:logout', handleLogout);
        return () => window.removeEventListener('careerpath:logout', handleLogout);
    }, []); // useState setters are stable — safe with empty deps

    const clearResults = () => {
        setPredictionResults(null);
        setUploadedFileName(null);
        setUploadedFile(null);
        setResumeText(null);
        setLearningRoadmapByPath(null);
        setCertificationDataByPath(null);
        setHistoryRecordId(null);
        sessionStorage.removeItem(SESSION_KEY);
    };

    return (
        <DashboardContext.Provider
            value={{
                predictionResults,
                setPredictionResults,
                uploadedFileName,
                setUploadedFileName,
                uploadedFile,
                setUploadedFile,
                resumeText,
                setResumeText,
                learningRoadmapByPath,
                setLearningRoadmapByPath,
                certificationDataByPath,
                setCertificationDataByPath,
                historyRecordId,
                setHistoryRecordId,
                clearResults,
            }}
        >
            {children}
        </DashboardContext.Provider>
    );
};

export default DashboardContext;
