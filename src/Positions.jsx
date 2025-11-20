import React, { useState, useEffect } from 'react';
import './Positions.css';

const API_URL = 'http://localhost:5001';

function Positions() {
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    title: '',
    department: '',
    location: '',
    openings: 1
  });

  useEffect(() => {
    fetchPositions();
  }, []);

  const fetchPositions = async () => {
    try {
      const response = await fetch(`${API_URL}/api/positions`);
      const data = await response.json();
      if (data.ok) {
        setPositions(data.positions);
      }
    } catch (error) {
      console.error('Error fetching positions:', error);
    } finally {
      setLoading(false);
    }
  };

  const createPosition = async (e) => {
    e.preventDefault();
    try {
      await fetch(`${API_URL}/api/positions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      setShowForm(false);
      setFormData({ title: '', department: '', location: '', openings: 1 });
      fetchPositions();
    } catch (error) {
      console.error('Error creating position:', error);
    }
  };

  const deletePosition = async (id) => {
    if (!window.confirm('Delete this position?')) return;
    try {
      await fetch(`${API_URL}/api/positions/${id}`, { method: 'DELETE' });
      fetchPositions();
    } catch (error) {
      console.error('Error deleting position:', error);
    }
  };

  if (loading) {
    return <div className="loading"><div className="spinner"></div><p>Loading positions...</p></div>;
  }

  return (
    <div className="positions-page">
      <div className="page-header">
        <div>
          <h1>Open Positions ({positions.length})</h1>
          <p>Manage job openings</p>
        </div>
        <button onClick={() => setShowForm(true)} className="btn btn-primary">
          <i className="fas fa-plus"></i> New Position
        </button>
      </div>

      <div className="positions-grid">
        {positions.map(pos => (
          <div key={pos.id} className="position-card">
            <div className="position-header">
              <h3>{pos.title}</h3>
              <span className={`badge badge-${pos.status}`}>{pos.status}</span>
            </div>
            <div className="position-info">
              <div><i className="fas fa-building"></i> {pos.department}</div>
              <div><i className="fas fa-map-marker-alt"></i> {pos.location}</div>
              <div><i className="fas fa-users"></i> {pos.openings} openings • {pos.filled} filled</div>
            </div>
            <div className="position-actions">
              <button onClick={() => deletePosition(pos.id)} className="btn btn-sm btn-danger">
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      {showForm && (
        <div className="modal-overlay" onClick={() => setShowForm(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>New Position</h2>
              <button className="modal-close" onClick={() => setShowForm(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <form onSubmit={createPosition}>
              <div className="form-group">
                <label>Title</label>
                <input
                  type="text"
                  className="form-control"
                  value={formData.title}
                  onChange={e => setFormData({...formData, title: e.target.value})}
                  required
                />
              </div>
              <div className="form-group">
                <label>Department</label>
                <input
                  type="text"
                  className="form-control"
                  value={formData.department}
                  onChange={e => setFormData({...formData, department: e.target.value})}
                />
              </div>
              <div className="form-group">
                <label>Location</label>
                <input
                  type="text"
                  className="form-control"
                  value={formData.location}
                  onChange={e => setFormData({...formData, location: e.target.value})}
                />
              </div>
              <div className="form-group">
                <label>Openings</label>
                <input
                  type="number"
                  className="form-control"
                  value={formData.openings}
                  onChange={e => setFormData({...formData, openings: parseInt(e.target.value)})}
                  min="1"
                />
              </div>
              <div style={{display: 'flex', gap: '1rem'}}>
                <button type="submit" className="btn btn-primary">Create</button>
                <button type="button" onClick={() => setShowForm(false)} className="btn btn-secondary">Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default Positions;
