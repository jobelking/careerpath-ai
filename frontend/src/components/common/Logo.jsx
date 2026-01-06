import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Logo.css';

const Logo = ({ variant = 'modern', className = '' }) => {
  const navigate = useNavigate();

  return (
    <div 
      className={`logo-container logo-${variant} ${className}`} 
      onClick={() => navigate('/')}
    >
      {/* Design 1: Modern (The default, clean look) */}
      {variant === 'modern' && (
        <>
          <span className="text-primary">CareerPath</span>
          <span className="text-accent">-AI</span>
        </>
      )}

      {/* Design 2: Tech (Futuristic, monospaced/geometric) */}
      {variant === 'tech' && (
        <>
          <span className="tech-bracket">&lt;</span>
          <span className="text-primary">CP</span>
          <span className="tech-slash">/</span>
          <span className="text-accent">AI</span>
          <span className="tech-bracket">&gt;</span>
        </>
      )}

      {/* Design 3: Classic (Editorial, trustworthy serif) */}
      {variant === 'classic' && (
        <>
          <span className="text-primary">Career</span>
          <span className="text-accent">Path.ai</span>
        </>
      )}
    </div>
  );
};

export default Logo;