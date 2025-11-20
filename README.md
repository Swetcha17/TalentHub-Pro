# TalentHub Pro - Setup Guide

## 🎯 Overview
TalentHub Pro is an advanced recruitment platform with a Python Flask backend and React frontend. This guide will help you get everything working properly.

## 📁 Project Structure

```
talenthub-pro/
├── parse_modified.py          # Backend server (Flask API)
├── package.json               # React dependencies
├── public/
│   └── index.html            # HTML template
├── src/
│   ├── App.jsx               # Main app component (routing)
│   ├── App.css               # Global styles
│   ├── Analytics.jsx         # Analytics page
│   ├── Analytics.css         # Analytics styles
│   ├── Candidates.jsx        # Candidates page
│   ├── Candidates.css        # Candidates styles
│   ├── Positions.jsx         # Positions page
│   ├── Positions.css         # Positions styles
│   └── index.js              # React entry point
└── build/                     # Production build (created after npm run build)
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install flask flask-cors PyMuPDF pdfplumber python-docx pytesseract pillow easyocr spacy
python -m spacy download en_core_web_sm
```

### Step 2: Build the React App

```bash
# Create production build
npm run build
```

This creates a `build/` folder with optimized static files.

### Step 3: Start the Backend Server

```bash
# Rename the modified parse file
mv parse_modified.py parse.py

# Start the server
python parse.py --web
```

The application will be available at: **http://localhost:5001**

## 🔄 Development Mode

### Option 1: Separate Dev Servers (Recommended for Development)

```bash
# Terminal 1 - Start React dev server
npm start
# Opens at http://localhost:3000

# Terminal 2 - Start Flask backend
python parse.py --web
# Runs at http://localhost:5001
```

The React dev server will proxy API requests to the Flask backend (configured in package.json).

### Option 2: Production Mode

```bash
# Build React app
npm run build

# Start Flask (serves both frontend and API)
python parse.py --web
# Everything at http://localhost:5001
```

## 📋 Features

### Analytics Page
- Total candidates overview
- Status breakdown visualization
- Pipeline stages
- Experience distribution
- Top skills chart
- Applications timeline

### Candidates Page
- Searchable candidate list
- Status filtering
- Inline status updates
- Resume downloads
- Duplicate removal
- Email links

### Positions Page
- Job openings management
- Create new positions
- Track openings vs filled
- Department organization

## 🔧 Configuration

### Backend (parse.py)
- Port: 5001
- Database: SQLite (resumes.db)
- Upload folder: ./uploads/
- Supported formats: PDF, DOCX, DOC, TXT, PNG, JPG

### Frontend (package.json)
- React version: 18.2.0
- React Router: 6.20.0
- API proxy: http://localhost:5001

## 📡 API Endpoints

### Candidates
- `GET /api/candidates` - List all candidates
- `POST /api/candidate/:id/update` - Update candidate status
- `GET /api/candidate/:id/download` - Download resume
- `POST /api/remove_duplicates` - Remove duplicate candidates

### Positions
- `GET /api/positions` - List all positions
- `POST /api/positions` - Create new position
- `DELETE /api/positions/:id` - Delete position

### Analytics
- `GET /api/analytics` - Get analytics data

### Search & Upload
- `POST /api/search` - Search resumes
- `POST /api/upload` - Upload single resume
- `POST /api/batch_upload` - Upload multiple resumes

## 🎨 Styling

The app uses:
- Custom CSS with CSS variables for theming
- Gradient backgrounds with animations
- Glassmorphism effects
- Responsive design (mobile-friendly)
- Font Awesome icons

## 🐛 Troubleshooting

### Issue: "Build folder not found"
**Solution**: Run `npm run build` before starting the Flask server in production mode.

### Issue: "Module not found" errors in React
**Solution**: Run `npm install` to install all dependencies.

### Issue: API requests failing
**Solution**: 
1. Check Flask server is running on port 5001
2. Verify CORS is enabled in parse.py
3. Check browser console for specific errors

### Issue: White screen on load
**Solution**:
1. Check browser console for errors
2. Verify all .jsx files are in the src/ folder
3. Rebuild the app: `npm run build`

### Issue: Styles not loading
**Solution**:
1. Ensure all .css files are in src/
2. Check imports in .jsx files
3. Clear browser cache and rebuild

## 📦 Dependencies

### Frontend
- react: ^18.2.0
- react-dom: ^18.2.0
- react-router-dom: ^6.20.0
- react-scripts: 5.0.1

### Backend
- Flask
- flask-cors
- PyMuPDF (fitz)
- pdfplumber
- python-docx
- pytesseract
- Pillow
- spacy
- easyocr (optional)

## 🔐 Production Deployment

1. Set environment variables:
   ```bash
   export FLASK_ENV=production
   ```

2. Build React app:
   ```bash
   npm run build
   ```

3. Use production WSGI server:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5001 parse:app
   ```

## 📝 Notes

- The backend stores data in SQLite database (`resumes.db`)
- Uploaded resumes are stored in `./uploads/` directory
- The app uses client-side routing (React Router)
- All API endpoints are prefixed with `/api/`

## 🤝 Support

If you encounter issues:
1. Check the terminal output for error messages
2. Check browser console for frontend errors
3. Verify all files are in the correct locations
4. Ensure all dependencies are installed

## 📄 License

This is a private project for recruitment management.
