import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import Search from './Search';
import Analytics from './Analytics';
import Candidates from './Candidates';
import Positions from './Positions';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <header className="header">
          <div className="header-content">
            <NavLink to="/" className="logo">
              <i className="fas fa-briefcase"></i>
              <span>TalentHub Pro</span>
            </NavLink>
            <nav className="nav-items">
              <NavLink to="/analytics" className="nav-item">
                <i className="fas fa-chart-line"></i> Analytics
              </NavLink>
              <NavLink to="/candidates" className="nav-item">
                <i className="fas fa-users"></i> Candidates
              </NavLink>
              <NavLink to="/positions" className="nav-item">
                <i className="fas fa-briefcase"></i> Positions
              </NavLink>
            </nav>
          </div>
        </header>

        <main className="main-content">
          <div className="container">
            <Routes>
              <Route path="/" element={<Search />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/candidates" element={<Candidates />} />
              <Route path="/positions" element={<Positions />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
