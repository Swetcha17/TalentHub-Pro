import React, { useState, useEffect, useRef } from 'react';
import './Search.css';

const API_URL =
  process.env.NODE_ENV === 'production'
    ? ''
    : 'http://localhost:5001';

function Search() {
  const [searchQuery, setSearchQuery] = useState('');
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [showModal, setShowModal] = useState(false);
  
  // Autocomplete
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [activeSuggestion, setActiveSuggestion] = useState(0);
  const searchRef = useRef(null);
  
  // Filters
  const [filters, setFilters] = useState({
    location: '',
    experience: [0, 15],
    skills: [],
    experienceLevel: [],
    education: [],
    workAuth: [],
    remote: []
  });

  const [skillInput, setSkillInput] = useState('');
  const [filterOptions, setFilterOptions] = useState({
    skills: [],
    locations: [],
    companies: []
  });

  useEffect(() => {
    fetchFilterOptions();
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const fetchFilterOptions = async () => {
    try {
      const response = await fetch(`${API_URL}/api/filters/options`);
      const data = await response.json();
      if (data.ok) {
        setFilterOptions(data);
      }
    } catch (err) {
      console.error('Error fetching filter options:', err);
    }
  };

  const fetchAutocomplete = async (query) => {
    if (!query || query.length < 2) {
      setSuggestions([]);
      return;
    }

    try {
      const response = await fetch(`${API_URL}/api/autocomplete?q=${encodeURIComponent(query)}&field=all`);
      const data = await response.json();
      setSuggestions(data);
      setShowSuggestions(true);
    } catch (err) {
      console.error('Autocomplete error:', err);
      setSuggestions([]);
    }
  };

  const handleSearchChange = (e) => {
    const value = e.target.value;
    setSearchQuery(value);
    fetchAutocomplete(value);
  };

  const handleSuggestionClick = (suggestion) => {
    setSearchQuery(suggestion);
    setShowSuggestions(false);
    handleSearch(suggestion);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveSuggestion(prev => 
        prev < suggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveSuggestion(prev => prev > 0 ? prev - 1 : 0);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (showSuggestions && suggestions[activeSuggestion]) {
        handleSuggestionClick(suggestions[activeSuggestion]);
      } else {
        handleSearch();
      }
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  const handleSearch = async (query = searchQuery) => {
    if (!query.trim() && filters.skills.length === 0 && !filters.location) {
      return;
    }
    
    setLoading(true);
    setSearched(true);
    
    try {
      const response = await fetch(`${API_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          location: filters.location,
          filters: filters
        })
      });
      
      const data = await response.json();
      if (Array.isArray(data)) {
        setCandidates(data);
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  };

  const addSkill = () => {
    if (skillInput.trim() && !filters.skills.includes(skillInput.trim())) {
      setFilters(prev => ({
        ...prev,
        skills: [...prev.skills, skillInput.trim()]
      }));
      setSkillInput('');
    }
  };

  const removeSkill = (skill) => {
    setFilters(prev => ({
      ...prev,
      skills: prev.skills.filter(s => s !== skill)
    }));
  };

  const clearSearch = () => {
    setSearchQuery('');
    setCandidates([]);
    setSearched(false);
    setFilters({
      location: '',
      experience: [0, 15],
      skills: [],
      experienceLevel: [],
      education: [],
      workAuth: [],
      remote: []
    });
  };

  const openCandidateModal = async (candidateId) => {
    try {
      const response = await fetch(`${API_URL}/api/candidate/${candidateId}/details`);
      const data = await response.json();
      if (data.ok) {
        setSelectedCandidate(data.candidate);
        setShowModal(true);
      }
    } catch (err) {
      console.error('Error fetching candidate details:', err);
    }
  };

  const updateCandidateStatus = async (candidateId, status) => {
    try {
      const response = await fetch(`${API_URL}/api/candidate/${candidateId}/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      });
      
      if (response.ok) {
        setCandidates(prev => prev.map(c => 
          c.id === candidateId ? { ...c, status } : c
        ));
        if (selectedCandidate && selectedCandidate.id === candidateId) {
          setSelectedCandidate(prev => ({ ...prev, status }));
        }
      }
    } catch (error) {
      console.error('Error updating status:', error);
    }
  };

  return (
    <div className="search-page">
      <div className="search-header">
        <h1>Find Perfect Candidates</h1>
        <p>Search through resumes with advanced AI-powered matching</p>
      </div>

      <div className="search-container">
        <div className="search-main">
          <div className="search-input-group" ref={searchRef}>
            <input
              type="text"
              placeholder="Search for candidates... (e.g., 'Python developer with 5 years experience')"
              value={searchQuery}
              onChange={handleSearchChange}
              onKeyDown={handleKeyDown}
              onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
              className="search-input"
            />
            {showSuggestions && suggestions.length > 0 && (
              <div className="autocomplete-dropdown">
                {suggestions.map((suggestion, index) => (
                  <div
                    key={index}
                    className={`autocomplete-item ${index === activeSuggestion ? 'active' : ''}`}
                    onClick={() => handleSuggestionClick(suggestion)}
                    onMouseEnter={() => setActiveSuggestion(index)}
                  >
                    <i className="fas fa-search"></i>
                    {suggestion}
                  </div>
                ))}
              </div>
            )}
            <button 
              onClick={() => handleSearch()} 
              disabled={loading || (!searchQuery.trim() && filters.skills.length === 0)}
              className="search-btn"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
            {searched && (
              <button onClick={clearSearch} className="clear-btn">
                Clear
              </button>
            )}
          </div>
        </div>

        <div className="filters-section">
          <h3>Filters</h3>
          
          <div className="filter-grid">
            <div className="filter-group">
              <label>Location</label>
              <input
                type="text"
                placeholder="City, State, or Remote"
                value={filters.location}
                onChange={(e) => setFilters(prev => ({...prev, location: e.target.value}))}
                className="filter-input"
                list="location-options"
              />
              <datalist id="location-options">
                {filterOptions.locations.map((loc, idx) => (
                  <option key={idx} value={loc} />
                ))}
              </datalist>
            </div>

            <div className="filter-group">
              <label>Required Skills</label>
              <div className="skills-input">
                <input
                  type="text"
                  placeholder="Add skill (e.g., Python, React)"
                  value={skillInput}
                  onChange={(e) => setSkillInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && addSkill()}
                  className="filter-input"
                  list="skill-options"
                />
                <datalist id="skill-options">
                  {filterOptions.skills.map((skill, idx) => (
                    <option key={idx} value={skill.value}>
                      ({skill.count} candidates)
                    </option>
                  ))}
                </datalist>
                <button onClick={addSkill} className="add-skill-btn">Add</button>
              </div>
              <div className="skills-tags">
                {filters.skills.map(skill => (
                  <span key={skill} className="skill-tag">
                    {skill}
                    <button onClick={() => removeSkill(skill)} className="remove-skill">×</button>
                  </span>
                ))}
              </div>
            </div>

            <div className="filter-group">
              <label>Years of Experience: {filters.experience[0]} - {filters.experience[1]}+ years</label>
              <input
                type="range"
                min="0"
                max="15"
                value={filters.experience[1]}
                onChange={(e) => setFilters(prev => ({
                  ...prev, 
                  experience: [0, parseInt(e.target.value)]
                }))}
                className="range-input"
              />
            </div>
          </div>
        </div>

        {searched && (
          <div className="search-results">
            <div className="results-header">
              <h3>Search Results ({candidates.length} candidates found)</h3>
            </div>
            
            {loading && (
              <div className="loading">
                <i className="fas fa-spinner fa-spin"></i>
                Searching candidates...
              </div>
            )}

            {!loading && candidates.length === 0 && (
              <div className="no-results">
                <i className="fas fa-search"></i>
                <p>No candidates found matching your criteria</p>
                <p>Try adjusting your search terms or filters</p>
              </div>
            )}

            {!loading && candidates.length > 0 && (
              <div className="candidates-grid">
                {candidates.map((candidate, index) => (
                  <div key={candidate.id || index} className="candidate-card">
                    <div className="candidate-header">
                      <h4>{candidate.name}</h4>
                      <span className="match-score">
                        {candidate.matchScore ? `${candidate.matchScore}% match` : ''}
                      </span>
                    </div>
                    
                    <div className="candidate-info">
                      <p className="title">{candidate.title}</p>
                      {candidate.location && (
                        <p className="location">
                          <i className="fas fa-map-marker-alt"></i>
                          {candidate.location}
                        </p>
                      )}
                      {candidate.experience && (
                        <p className="experience">
                          <i className="fas fa-briefcase"></i>
                          {candidate.experience} years experience
                        </p>
                      )}
                      {candidate.email && (
                        <p className="email">
                          <i className="fas fa-envelope"></i>
                          {candidate.email}
                        </p>
                      )}
                    </div>

                    {candidate.skills && candidate.skills.length > 0 && (
                      <div className="candidate-skills">
                        {candidate.skills.slice(0, 5).map((skill, idx) => (
                          <span key={idx} className="skill-badge">{skill}</span>
                        ))}
                        {candidate.skills.length > 5 && (
                          <span className="skill-badge">+{candidate.skills.length - 5}</span>
                        )}
                      </div>
                    )}

                    {candidate.summary && (
                      <div className="candidate-summary">
                        <p>{candidate.summary.substring(0, 150)}...</p>
                      </div>
                    )}

                    <div className="candidate-actions">
                      <button 
                        className="action-btn primary"
                        onClick={() => openCandidateModal(candidate.id)}
                      >
                        <i className="fas fa-eye"></i>
                        View Profile
                      </button>
                      <select
                        className="action-btn secondary"
                        value={candidate.status || 'new'}
                        onChange={(e) => {
                          e.stopPropagation();
                          updateCandidateStatus(candidate.id, e.target.value);
                        }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <option value="new">New</option>
                        <option value="shortlisted">Shortlist</option>
                        <option value="interviewing">Interviewing</option>
                        <option value="offered">Offered</option>
                        <option value="hired">Hired</option>
                        <option value="rejected">Rejected</option>
                      </select>
                      <a
                        href={`${API_URL}/api/candidate/${candidate.id}/download`}
                        download
                        className="action-btn secondary"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <i className="fas fa-download"></i>
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {!searched && (
          <div className="search-tips">
            <h3>Search Tips</h3>
            <div className="tips-grid">
              <div className="tip">
                <i className="fas fa-lightbulb"></i>
                <h4>Be Specific</h4>
                <p>Try "Senior React developer with Node.js experience" instead of just "developer"</p>
              </div>
              <div className="tip">
                <i className="fas fa-filter"></i>
                <h4>Use Filters</h4>
                <p>Combine search terms with location, experience level, and skill filters</p>
              </div>
              <div className="tip">
                <i className="fas fa-search-plus"></i>
                <h4>Try Different Terms</h4>
                <p>Search for both "JavaScript" and "JS", "Machine Learning" and "ML"</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {showModal && selectedCandidate && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h2>{selectedCandidate.name}</h2>
                <p className="modal-subtitle">{selectedCandidate.title}</p>
              </div>
              <button 
                className="modal-close"
                onClick={() => setShowModal(false)}
              >
                <i className="fas fa-times"></i>
              </button>
            </div>
            
            <div className="modal-body">
              <div className="modal-section">
                <h3><i className="fas fa-user"></i> Contact Information</h3>
                <div className="info-grid">
                  {selectedCandidate.email && (
                    <div className="info-item">
                      <i className="fas fa-envelope"></i>
                      <span>{selectedCandidate.email}</span>
                    </div>
                  )}
                  {selectedCandidate.phone && (
                    <div className="info-item">
                      <i className="fas fa-phone"></i>
                      <span>{selectedCandidate.phone}</span>
                    </div>
                  )}
                  {selectedCandidate.location && (
                    <div className="info-item">
                      <i className="fas fa-map-marker-alt"></i>
                      <span>{selectedCandidate.location}</span>
                    </div>
                  )}
                  <div className="info-item">
                    <i className="fas fa-briefcase"></i>
                    <span>{selectedCandidate.experience} years experience</span>
                  </div>
                </div>
              </div>

              {selectedCandidate.summary && (
                <div className="modal-section">
                  <h3><i className="fas fa-file-alt"></i> Summary</h3>
                  <p>{selectedCandidate.summary}</p>
                </div>
              )}

              {selectedCandidate.skills && selectedCandidate.skills.length > 0 && (
                <div className="modal-section">
                  <h3><i className="fas fa-code"></i> Skills</h3>
                  <div className="skills-list">
                    {selectedCandidate.skills.map((skill, idx) => (
                      <span key={idx} className="skill-badge">{skill}</span>
                    ))}
                  </div>
                </div>
              )}

              {selectedCandidate.education && selectedCandidate.education.length > 0 && (
                <div className="modal-section">
                  <h3><i className="fas fa-graduation-cap"></i> Education</h3>
                  {selectedCandidate.education.map((edu, idx) => (
                    <div key={idx} className="education-item">
                      <strong>{edu.degree}</strong>
                      {edu.institution && <p>{edu.institution}</p>}
                      {edu.year && <p>{edu.year}</p>}
                    </div>
                  ))}
                </div>
              )}

              {selectedCandidate.companies && selectedCandidate.companies.length > 0 && (
                <div className="modal-section">
                  <h3><i className="fas fa-building"></i> Experience</h3>
                  <div className="companies-list">
                    {selectedCandidate.companies.map((company, idx) => (
                      <span key={idx} className="company-badge">{company}</span>
                    ))}
                  </div>
                </div>
              )}

              <div className="modal-actions">
                <select
                  value={selectedCandidate.status || 'new'}
                  onChange={(e) => updateCandidateStatus(selectedCandidate.id, e.target.value)}
                  className="status-select"
                >
                  <option value="new">New</option>
                  <option value="shortlisted">Shortlisted</option>
                  <option value="interviewing">Interviewing</option>
                  <option value="offered">Offered</option>
                  <option value="hired">Hired</option>
                  <option value="rejected">Rejected</option>
                </select>
                <a
                  href={`${API_URL}/api/candidate/${selectedCandidate.id}/download`}
                  download
                  className="btn btn-primary"
                >
                  <i className="fas fa-download"></i>
                  Download Resume
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Search;
