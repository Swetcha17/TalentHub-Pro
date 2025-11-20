# 🚀 QUICK START - TalentHub Pro

## ⚡ Super Fast Setup (3 Commands)

```bash
# 1. Make setup script executable and run it
chmod +x setup.sh && ./setup.sh

# 2. Rename the backend file
mv parse_modified.py parse.py

# 3. Start the server
python parse.py --web
```

**That's it!** Open http://localhost:5001 in your browser 🎉

---

## 📋 What Just Happened?

1. ✅ Installed Node.js and Python dependencies
2. ✅ Built the React frontend app
3. ✅ Started Flask backend server (serves React + API)

---

## 🎯 Your App Structure

```
├── parse.py              ← Backend (Flask server + API)
├── package.json          ← React configuration
├── src/                  ← Your React components
│   ├── App.jsx          ← Main app (routing)
│   ├── Analytics.jsx    ← Analytics page
│   ├── Candidates.jsx   ← Candidates page
│   └── Positions.jsx    ← Positions page
└── build/               ← Built React app (served by Flask)
```

---

## 🔗 How It Works

```
Browser (localhost:5001)
    ↓
Flask Backend (parse.py)
    ├─→ Serves React App (from build/)
    └─→ Handles API requests (/api/*)
        └─→ SQLite Database (resumes.db)
```

---

## 📱 Using The App

1. **Analytics** - View dashboard with charts and metrics
2. **Candidates** - Search, filter, and manage candidates
3. **Positions** - Create and manage job openings

---

## 🛠️ Common Commands

```bash
# Start production server
python parse.py --web

# Development mode (with hot reload)
npm start                # Terminal 1 - React dev server
python parse.py --web    # Terminal 2 - Backend

# Rebuild after changes
npm run build
```

---

## 🔥 Next Steps

1. 📤 Upload resumes through the UI
2. 🔍 Try searching for candidates
3. 📊 Check the analytics dashboard
4. 💼 Create some job positions

---

## ❓ Need Help?

- **Detailed Docs**: See `README.md`
- **Project Structure**: See `PROJECT_STRUCTURE.md`
- **Issues**: Check browser console (F12) and terminal output

---

## 🎨 Key Files to Edit

| Want to... | Edit this file... |
|------------|-------------------|
| Change backend logic | `parse.py` |
| Add a new page | `src/App.jsx` (add route) |
| Modify analytics | `src/Analytics.jsx` |
| Change candidates view | `src/Candidates.jsx` |
| Adjust positions | `src/Positions.jsx` |
| Update styles | Respective `.css` files |

---

**You're all set! Happy recruiting! 🎉**
