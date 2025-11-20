# 📁 TalentHub Pro - Complete Project Structure

## File Organization

```
TalentHub-Pro/
│
├── 📄 parse_modified.py          ⭐ BACKEND SERVER (Use this instead of parse.py)
│   └── Flask server with API endpoints
│   └── Serves React build folder
│   └── Resume parsing and search logic
│
├── 📄 package.json               ⭐ React dependencies configuration
│   └── Defines npm scripts (start, build)
│   └── Lists React dependencies
│   └── Configures proxy to backend (port 5001)
│
├── 📄 setup.sh                   ⭐ QUICK SETUP SCRIPT (Run this first!)
│   └── Automated installation script
│   └── Installs all dependencies
│   └── Builds the React app
│
├── 📄 README.md                  📖 Main documentation
│
├── 📁 public/                    🌐 React public assets
│   └── index.html                   HTML template
│
├── 📁 src/                       ⚛️ React source code
│   ├── App.jsx                      Main app with routing ⭐
│   ├── App.css                      Global styles
│   ├── Analytics.jsx                Analytics dashboard page
│   ├── Analytics.css                Analytics styles
│   ├── Candidates.jsx               Candidates management page
│   ├── Candidates.css               Candidates styles
│   ├── Positions.jsx                Job positions page
│   ├── Positions.css                Positions styles
│   └── index.js                     React entry point
│
└── 📁 build/                     📦 Production build (created by npm run build)
    └── Static files served by Flask
    └── Optimized for production

```

## 🔗 How They Link Together

```
┌─────────────────────────────────────────────────────────────┐
│                         USER BROWSER                          │
│                    http://localhost:5001                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   FLASK BACKEND (parse_modified.py)           │
│                                                               │
│  Routes:                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ GET /              → Serves React App (index.html)   │   │
│  │ GET /candidates    → Serves React App                │   │
│  │ GET /positions     → Serves React App                │   │
│  │ GET /analytics     → Serves React App                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  API Routes:                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ GET  /api/candidates        → List candidates        │   │
│  │ POST /api/candidates/:id    → Update candidate       │   │
│  │ GET  /api/analytics         → Get analytics data     │   │
│  │ GET  /api/positions         → List positions         │   │
│  │ POST /api/positions         → Create position        │   │
│  │ POST /api/upload            → Upload resumes         │   │
│  │ POST /api/search            → Search resumes         │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
                   ┌────────────────┐
                   │   SQLite DB    │
                   │  resumes.db    │
                   └────────────────┘
```

## 🎯 React Component Flow

```
                    ┌─────────────┐
                    │   index.js  │  ← Entry point
                    └──────┬──────┘
                           │
                           ↓
                    ┌─────────────┐
                    │   App.jsx   │  ← Main app with routing
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ↓               ↓               ↓
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │Analytics │   │Candidates│   │Positions │
    │   .jsx   │   │   .jsx   │   │   .jsx   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         ↓              ↓              ↓
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │Analytics │   │Candidates│   │Positions │
    │   .css   │   │   .css   │   │   .css   │
    └──────────┘   └──────────┘   └──────────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
                   API Calls to
                   Flask Backend
```

## 🚀 Quick Start Commands

### 1️⃣ Automatic Setup (Recommended)
```bash
chmod +x setup.sh
./setup.sh
```

### 2️⃣ Manual Setup
```bash
# Install dependencies
npm install
pip install flask flask-cors PyMuPDF pdfplumber python-docx pytesseract pillow spacy

# Build React app
npm run build

# Rename the backend file
mv parse_modified.py parse.py

# Start server
python parse.py --web
```

## 🔄 Data Flow Example

### Viewing Candidates:
```
User clicks "Candidates" in navbar
    ↓
React Router loads Candidates.jsx
    ↓
Candidates.jsx calls: fetch('http://localhost:5001/api/candidates')
    ↓
Flask route /api/candidates executes
    ↓
Query SQLite database for candidates
    ↓
Return JSON data to frontend
    ↓
Candidates.jsx displays data in table
```

### Uploading Resume:
```
User uploads file via UI
    ↓
POST request to /api/upload
    ↓
Flask saves file to ./uploads/
    ↓
parse.py extracts text and data
    ↓
Store in SQLite database
    ↓
Return success response
    ↓
UI updates candidate list
```

## 🎨 Styling System

```
App.css (Global)
    ↓
├── CSS Variables (:root)
├── Base styles (body, fonts)
├── Header & Navigation
├── Layout (containers, grids)
├── Common components (buttons, cards)
└── Responsive breakpoints

Component.css (Specific)
    ↓
├── Page-specific styles
├── Component layouts
└── Custom elements
```

## 📦 Build Process

```
Source Files (src/)
    ↓
npm run build
    ↓
Webpack/Babel Processing
    ↓
Build Folder (build/)
    ├── index.html
    ├── static/
    │   ├── js/
    │   │   └── main.[hash].js  ← All React code bundled
    │   └── css/
    │       └── main.[hash].css ← All styles bundled
    └── asset-manifest.json
```

## 🔍 Important Files Explained

| File | Purpose | Edit? |
|------|---------|-------|
| `parse_modified.py` | Backend server & API | Yes - for backend logic |
| `src/App.jsx` | Main app routing | Yes - to add new pages |
| `src/Analytics.jsx` | Analytics page | Yes - to customize analytics |
| `src/Candidates.jsx` | Candidates page | Yes - to customize candidate view |
| `src/Positions.jsx` | Positions page | Yes - to customize positions |
| `package.json` | Dependencies & scripts | Rarely - only for new packages |
| `public/index.html` | HTML template | Rarely - only for meta tags |

## 🎯 Next Steps After Setup

1. ✅ Run `./setup.sh`
2. ✅ Start server: `python parse.py --web`
3. ✅ Open browser: `http://localhost:5001`
4. 📤 Upload some resumes
5. 📊 View analytics
6. 🔍 Search candidates
7. 💼 Create positions

## 💡 Development Tips

- **Hot Reload**: Use `npm start` for live React updates during development
- **Backend Changes**: Restart Flask server after editing parse.py
- **Styling Changes**: Edit .css files and rebuild
- **New Components**: Add to src/, import in App.jsx
- **API Changes**: Edit parse_modified.py backend routes

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Port 5001 in use | Change port in parse.py and package.json proxy |
| Build folder missing | Run `npm run build` |
| Module not found | Run `npm install` |
| API 404 errors | Check Flask server is running |
| Blank page | Check browser console for errors |

---

**Ready to start? Run: `./setup.sh`** 🚀
