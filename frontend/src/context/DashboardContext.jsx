import React, { createContext, useContext, useState } from 'react';

const DashboardContext = createContext();

export const useDashboard = () => {
    const context = useContext(DashboardContext);
    if (!context) {
        throw new Error('useDashboard must be used within a DashboardProvider');
    }
    return context;
};

export const DashboardProvider = ({ children }) => {
    const [predictionResults, setPredictionResults] = useState(null);
    const [uploadedFileName, setUploadedFileName] = useState(null);
    const [uploadedFile, setUploadedFile] = useState(null);
    const [resumeText, setResumeText] = useState(null);
    const [learningRoadmap, setLearningRoadmap] = useState(null);
    const [certificationData, setCertificationData] = useState(null);

    const clearResults = () => {
        setPredictionResults(null);
        setUploadedFileName(null);
        setUploadedFile(null);
        setResumeText(null);
        setLearningRoadmap(null);
        setCertificationData(null);
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
