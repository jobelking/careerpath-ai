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
    const [learningRoadmap, setLearningRoadmap] = useState(persisted.learningRoadmap ?? null);
    const [certificationData, setCertificationData] = useState(persisted.certificationData ?? null);

    // Sync serializable state to sessionStorage whenever it changes
    useEffect(() => {
        try {
            sessionStorage.setItem(SESSION_KEY, JSON.stringify({
                predictionResults,
                uploadedFileName,
                resumeText,
                learningRoadmap,
                certificationData,
            }));
        } catch {
            // Ignore storage quota errors
        }
    }, [predictionResults, uploadedFileName, resumeText, learningRoadmap, certificationData]);

    const clearResults = () => {
        setPredictionResults(null);
        setUploadedFileName(null);
        setUploadedFile(null);
        setResumeText(null);
        setLearningRoadmap(null);
        setCertificationData(null);
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
                learningRoadmap,
                setLearningRoadmap,
                certificationData,
                setCertificationData,
                clearResults,
            }}
        >
            {children}
        </DashboardContext.Provider>
    );
};

export default DashboardContext;
