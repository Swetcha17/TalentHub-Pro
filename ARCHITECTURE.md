# TalentHub Pro - Architecture Overview

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (localhost:5001)                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    React Frontend                      │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │           App.jsx (Main Router)                  │ │  │
│  │  │  ┌──────────┐  ┌───────────┐  ┌──────────┐     │ │  │
│  │  │  │Analytics │  │Candidates │  │Positions │     │ │  │
│  │  │  │  Page    │  │   Page    │  │   Page   │     │ │  │
│  │  │  └────┬─────┘  └─────┬─────┘  └────┬─────┘     │ │  │
│  │  └───────┼───────────────┼─────────────┼───────────┘ │  │
│  └──────────┼───────────────┼─────────────┼─────────────┘  │
└─────────────┼───────────────┼─────────────┼────────────────┘
              │               │             │
              │   HTTP/REST   │             │
              │   API Calls   │             │
              ▼               ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Backend (parse.py)                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  API Endpoints                         │  │
│  │  • GET  /api/analytics     (Analytics data)           │  │
│  │  • GET  /api/candidates    (All candidates)           │  │
│  │  • POST /api/candidate/:id/update (Update status)     │  │
│  │  • GET  /api/positions     (Job positions)            │  │
│  │  • POST /api/upload        (Upload resumes)           │  │
│  │  • POST /api/search        (Search resumes)           │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │         ResumeSearchSystem (Core Logic)               │  │
│  │  • Resume parsing (PDF, DOCX, images)                 │  │
│  │  • AI semantic search                                 │  │
│  │  • Candidate deduplication                            │  │
│  │  • Skills & experience extraction                     │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │              Data Storage Layer                        │  │
│  │  • resumes_db.json    (Candidate data)                │  │
│  │  • positions_db.json  (Job positions)                 │  │
│  │  • uploads/           (Resume files)                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Request Flow Example

### Example: Viewing Candidates Page

```
1. User clicks "Candidates" in navigation
   ↓
2. App.jsx updates state to show Candidates component
   ↓
3. Candidates.jsx mounts and calls useEffect()
   ↓
4. Sends GET request: fetch('http://localhost:5001/api/candidates')
   ↓
5. Flask route @app.get('/api/candidates') receives request
   ↓
6. parse.py queries system.resumes from resumes_db.json
   ↓
7. Returns JSON: {ok: true, candidates: [...]}
   ↓
8. Candidates.jsx receives data and updates state
   ↓
9. Component renders table with candidate data
```

### Example: Updating Candidate Status

```
1. User changes dropdown to "Shortlisted"
   ↓
2. Candidates.jsx calls updateCandidateStatus(id, 'shortlisted')
   ↓
3. Sends POST: fetch('/api/candidate/123/update', {status: 'shortlisted'})
   ↓
4. Flask route @app.post('/api/candidate/<id>/update') receives request
   ↓
5. Updates candidate in system.resumes
   ↓
6. Saves to resumes_db.json
   ↓
7. Returns JSON: {ok: true}
   ↓
8. Candidates.jsx updates local state
   ↓
9. UI shows updated status immediately
```

## 📦 Component Structure

```
src/
├── App.jsx                 # Main application component
│   ├── Header             # Navigation bar
│   ├── Router Logic       # Page switching
│   └── Container          # Page wrapper
│
├── Analytics.jsx          # Analytics dashboard
│   ├── Metrics Grid       # Total, Shortlisted, etc.
│   ├── Status Breakdown   # Bar charts
│   ├── Experience Dist    # Experience visualization
│   └── Skills Chart       # Top skills
│
├── Candidates.jsx         # Candidates management
│   ├── Search Bar         # Filter candidates
│   ├── Status Filter      # Dropdown filter
│   ├── Candidates Table   # Data table
│   └── Action Buttons     # Download, update
│
└── Positions.jsx          # Job positions
    ├── Position Cards     # Job listings
    ├── Create Form        # Add new position
    └── Action Buttons     # Delete position
```

## 🎨 Styling Architecture

```
CSS Files:
├── App.css              # Global styles, layout, common components
├── Analytics.css        # Analytics-specific styles
├── Candidates.css       # Candidates page styles
└── Positions.css        # Positions page styles

Style Hierarchy:
1. App.css defines global variables (--primary, --secondary, etc.)
2. Component CSS files use these variables
3. All components share common classes (btn, card, badge, etc.)
```

## 🔌 API Endpoints Reference

| Component | Endpoint | Purpose |
|-----------|----------|---------|
| Analytics | `/api/analytics` | Get dashboard metrics |
| Candidates | `/api/candidates` | Get all candidates |
| Candidates | `/api/candidate/:id/update` | Update candidate |
| Candidates | `/api/remove_duplicates` | Remove duplicates |
| Positions | `/api/positions` | Get/Create positions |
| Positions | `/api/positions/:id` | Delete position |

## 🚀 Build & Deployment Flow

```
Development:
src/*.jsx + src/*.css
    ↓
npm run build
    ↓
build/ folder
    ├── static/
    │   ├── js/main.[hash].js
    │   └── css/main.[hash].css
    └── index.html

Production:
parse.py serves files from build/
    ↓
Browser requests localhost:5001
    ↓
Flask returns build/index.html
    ↓
Browser loads JS/CSS from build/static/
    ↓
React app initializes
    ↓
Components make API calls back to Flask
```

## 🎯 Key Integration Points

1. **package.json proxy**: `"proxy": "http://localhost:5001"`
   - Redirects API calls from React dev server to Flask

2. **parse.py routes**: Modified to serve build/ folder
   ```python
   @app.route('/')
   def index():
       return send_from_directory('build', 'index.html')
   ```

3. **API_URL constant**: Set in each component
   ```javascript
   const API_URL = 'http://localhost:5001';
   ```

4. **CORS**: Flask allows all origins (dev only)
   ```python
   @app.after_request
   def after_request(response):
       response.headers.add('Access-Control-Allow-Origin', '*')
   ```

## ✅ Integration Checklist

- [x] App.jsx created with routing logic
- [x] All components properly imported
- [x] CSS files linked to components
- [x] parse.py updated to serve build folder
- [x] Flask imports include send_from_directory
- [x] API endpoints match component expectations
- [x] Build folder configured in route handlers
- [x] Error handling for missing build folder
