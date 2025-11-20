import React, { useState, useEffect } from 'react';
import './Candidates.css';

const API_URL = 'http://localhost:5001';

function Candidates() {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    fetchCandidates();
  }, []);

  const fetchCandidates = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/candidates`);
      const data = await response.json();
      if (data.ok) {
        setCandidates(data.candidates);
      }
    } catch (error) {
      console.error('Error fetching candidates:', error);
    } finally {
      setLoading(false);
    }
  };

  const searchCandidates = async () => {
    if (!searchTerm.trim()) {
      fetchCandidates();
      return;
    }

    setSearching(true);
    try {
      const response = await fetch(`${API_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchTerm })
      });
      const data = await response.json();
      
      // The search API returns array of candidates directly
      if (Array.isArray(data)) {
        setCandidates(data);
      }
    } catch (error) {
      console.error('Error searching candidates:', error);
    } finally {
      setSearching(false);
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
        // Update local state
        setCandidates(prev => prev.map(c => 
          c.id === candidateId ? { ...c, status } : c
        ));
      }
    } catch (error) {
      console.error('Error updating status:', error);
    }
  };

  const removeDuplicates = async () => {
    if (!window.confirm('Remove duplicate candidates?')) return;
    
    try {
      const response = await fetch(`${API_URL}/api/remove_duplicates`, {
        method: 'POST'
      });
      const data = await response.json();
      window.alert(data.message);
      fetchCandidates();
    } catch (error) {
      console.error('Error removing duplicates:', error);
    }
  };

  const handleFileUpload = async (event) => {
    event.stopPropagation();
    event.preventDefault();
    
    const files = event.target.files;
    console.log('Files selected:', files ? files.length : 0);
    
    if (!files || files.length === 0) {
      console.log('No files selected');
      return;
    }
    
    if (uploading) {
      console.log('Upload already in progress, skipping');
      return;
    }

    setUploading(true);
    setUploadProgress(`Uploading ${files.length} file(s)...`);

    try {
      const formData = new FormData();
      
      // Append each file
      for (let i = 0; i < files.length; i++) {
        console.log(`Adding file ${i + 1}: ${files[i].name}`);
        formData.append('files', files[i], files[i].name);
      }

      console.log('Sending upload request...');
      const response = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      console.log('Upload response:', data);
      
      if (data.ok) {
        setUploadProgress(`Upload complete! ${data.queued} files queued for processing. Refreshing...`);
        event.target.value = '';
        setTimeout(() => {
          fetchCandidates();
          setUploading(false);
          setUploadProgress('');
        }, 2000);
      } else {
        setUploadProgress('Upload failed: ' + (data.message || 'Unknown error'));
        event.target.value = '';
        setTimeout(() => {
          setUploading(false);
          setUploadProgress('');
        }, 3000);
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      setUploadProgress('Upload failed. Please try again.');
      event.target.value = '';
      setTimeout(() => {
        setUploading(false);
        setUploadProgress('');
      }, 3000);
    }
  };

  const filteredCandidates = candidates.filter(c => {
    const matchesSearch = c.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         c.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         (c.email && c.email.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesStatus = statusFilter === 'all' || c.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p>Loading candidates...</p>
      </div>
    );
  }

  return (
    <div className="candidates-page">
      <div className="page-header">
        <div>
          <h1>All Candidates ({candidates.length})</h1>
          <p>Manage and track all candidates in your database</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
            <i className="fas fa-upload"></i>
            {uploading ? 'Uploading...' : 'Upload Resumes'}
            <input
              type="file"
              multiple
              accept=".pdf,.doc,.docx,.txt"
              onChange={handleFileUpload}
              disabled={uploading}
              style={{ display: 'none' }}
            />
          </label>
          <button onClick={removeDuplicates} className="btn btn-secondary">
            <i className="fas fa-trash-alt"></i>
            Remove Duplicates
          </button>
        </div>
      </div>

      {uploadProgress && (
        <div style={{
          background: 'rgba(99, 102, 241, 0.1)',
          border: '1px solid #6366f1',
          color: '#4f46e5',
          padding: '1rem',
          borderRadius: '8px',
          marginBottom: '1rem',
          textAlign: 'center'
        }}>
          <i className="fas fa-spinner fa-spin"></i> {uploadProgress}
        </div>
      )}

      <div className="filters-bar">
        <div className="search-box">
          <i className="fas fa-search"></i>
          <input
            type="text"
            placeholder="Search by name, title, email, or skills..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && searchCandidates()}
          />
          <button 
            onClick={searchCandidates}
            disabled={searching}
            className="btn btn-primary btn-sm"
            style={{ marginLeft: '0.5rem' }}
          >
            {searching ? 'Searching...' : 'Search'}
          </button>
          {searchTerm && (
            <button 
              onClick={() => { setSearchTerm(''); fetchCandidates(); }}
              className="btn btn-secondary btn-sm"
              style={{ marginLeft: '0.5rem' }}
            >
              Clear
            </button>
          )}
        </div>

        <select 
          value={statusFilter} 
          onChange={(e) => setStatusFilter(e.target.value)}
          className="form-control"
        >
          <option value="all">All Status</option>
          <option value="new">New</option>
          <option value="shortlisted">Shortlisted</option>
          <option value="rejected">Not Interested</option>
          <option value="interviewing">Interviewing</option>
          <option value="offered">Offered</option>
          <option value="hired">Hired</option>
        </select>
      </div>

      <div className="table-card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Title</th>
                <th>Email</th>
                <th>Experience</th>
                <th>Skills</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredCandidates.map((candidate) => (
                <tr key={candidate.id}>
                  <td>
                    <div className="candidate-name">
                      <strong>{candidate.name}</strong>
                      {candidate.location && (
                        <div className="text-muted">
                          <i className="fas fa-map-marker-alt"></i> {candidate.location}
                        </div>
                      )}
                    </div>
                  </td>
                  <td>{candidate.title}</td>
                  <td>
                    {candidate.email && (
                      <a href={`mailto:${candidate.email}`}>{candidate.email}</a>
                    )}
                  </td>
                  <td>{candidate.experience} years</td>
                  <td>
                    <div className="skills-cell">
                      {candidate.skills.slice(0, 3).map((skill, idx) => (
                        <span key={idx} className="skill-tag">{skill}</span>
                      ))}
                      {candidate.skills.length > 3 && (
                        <span className="skill-tag">+{candidate.skills.length - 3}</span>
                      )}
                    </div>
                  </td>
                  <td>
                    <select
                      value={candidate.status || 'new'}
                      onChange={(e) => updateCandidateStatus(candidate.id, e.target.value)}
                      className="status-select"
                    >
                      <option value="new">New</option>
                      <option value="shortlisted">Shortlisted</option>
                      <option value="rejected">Not Interested</option>
                      <option value="interviewing">Interviewing</option>
                      <option value="offered">Offered</option>
                      <option value="hired">Hired</option>
                    </select>
                  </td>
                  <td>
                    <div className="action-buttons">
                      <a
                        href={`${API_URL}/api/candidate/${candidate.id}/download`}
                        download
                        className="btn btn-sm btn-secondary"
                        title="Download Resume"
                      >
                        <i className="fas fa-download"></i>
                      </a>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {filteredCandidates.length === 0 && (
            <div className="empty-state">
              <i className="fas fa-users"></i>
              <p>No candidates found</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Candidates;
