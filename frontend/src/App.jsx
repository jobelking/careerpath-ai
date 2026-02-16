import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { DashboardProvider } from './context/DashboardContext';
import LandingPage from './pages/LandingPage/LandingPage';
import Dashboard from './pages/Dashboard/Dashboard';
import Learnmore from './pages/Learnmore/learnmore';

import './App.css';

function App() {
  return (
    <DashboardProvider>
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/learnmore" element={<Learnmore />} />
        </Routes>
      </Router>
    </DashboardProvider>
  );
}

export default App;
