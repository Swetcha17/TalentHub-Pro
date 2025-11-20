import React, { useState, useEffect } from 'react';
import './Analytics.css';

const API_URL = 'http://localhost:5001';

function Analytics() {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      const response = await fetch(`${API_URL}/api/analytics`);
      const data = await response.json();
      if (data.ok) {
        setAnalytics(data);
      }
    } catch (error) {
      console.error('Error fetching analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p>Loading analytics...</p>
      </div>
    );
  }

  if (!analytics) {
    return <div className="error">Failed to load analytics</div>;
  }

  return (
    <div className="analytics-page">
      <div className="page-header">
        <h1>Analytics Dashboard</h1>
        <p>Insights and metrics for your recruitment pipeline</p>
      </div>

      <div className="metrics-grid">
        <div className="stat-card">
          <i className="fas fa-users"></i>
          <div className="value">{analytics.total_candidates}</div>
          <div className="label">Total Candidates</div>
        </div>

        <div className="stat-card">
          <i className="fas fa-star"></i>
          <div className="value">{analytics.status_breakdown.shortlisted || 0}</div>
          <div className="label">Shortlisted</div>
        </div>

        <div className="stat-card">
          <i className="fas fa-comments"></i>
          <div className="value">{analytics.status_breakdown.interviewing || 0}</div>
          <div className="label">Interviewing</div>
        </div>

        <div className="stat-card">
          <i className="fas fa-check-circle"></i>
          <div className="value">{analytics.status_breakdown.hired || 0}</div>
          <div className="label">Hired</div>
        </div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <h3>Status Breakdown</h3>
          <div className="breakdown-list">
            {Object.entries(analytics.status_breakdown).map(([status, count]) => (
              <div key={status} className="breakdown-item">
                <div className="breakdown-label">
                  <span className={`badge badge-${status}`}>{status}</span>
                  <span>{count} candidates</span>
                </div>
                <div className="breakdown-bar">
                  <div 
                    className="breakdown-fill"
                    style={{
                      width: `${(count / analytics.total_candidates) * 100}%`,
                      background: getStatusColor(status)
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h3>Pipeline Stages</h3>
          <div className="breakdown-list">
            {Object.entries(analytics.stage_breakdown).map(([stage, count]) => (
              <div key={stage} className="breakdown-item">
                <div className="breakdown-label">
                  <span>{stage}</span>
                  <span>{count}</span>
                </div>
                <div className="breakdown-bar">
                  <div 
                    className="breakdown-fill"
                    style={{
                      width: `${(count / analytics.total_candidates) * 100}%`
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Experience Distribution</h3>
        <div className="exp-distribution">
          {Object.entries(analytics.experience_distribution).map(([range, count]) => (
            <div key={range} className="exp-item">
              <div className="exp-label">{range} years</div>
              <div className="exp-bar">
                <div 
                  className="exp-fill"
                  style={{width: `${(count / analytics.total_candidates) * 100}%`}}
                />
              </div>
              <div className="exp-count">{count}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Top Skills</h3>
        <div className="skills-chart">
          {analytics.top_skills.slice(0, 15).map((item) => (
            <div key={item.skill} className="skill-bar-item">
              <div className="skill-name">{item.skill}</div>
              <div className="skill-bar">
                <div 
                  className="skill-fill"
                  style={{width: `${(item.count / analytics.total_candidates) * 100}%`}}
                />
              </div>
              <div className="skill-count">{item.count}</div>
            </div>
          ))}
        </div>
      </div>

      {analytics.applications_timeline.length > 0 && (
        <div className="card">
          <h3>Applications Timeline (Last 30 Days)</h3>
          <div className="timeline">
            {analytics.applications_timeline.map((item) => (
              <div key={item.date} className="timeline-item">
                <div className="timeline-date">{new Date(item.date).toLocaleDateString('en-US', {month: 'short', day: 'numeric'})}</div>
                <div className="timeline-bar">
                  <div 
                    className="timeline-fill"
                    style={{height: `${Math.min(100, (item.count / 10) * 100)}%`}}
                  />
                </div>
                <div className="timeline-count">{item.count}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function getStatusColor(status) {
  const colors = {
    new: '#3b82f6',
    shortlisted: '#10b981',
    interviewing: '#f59e0b',
    offered: '#8b5cf6',
    hired: '#06b6d4',
    rejected: '#ef4444'
  };
  return colors[status] || '#6b7280';
}

export default Analytics;
