# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 1. "Build folder not found" Error

**Error Message:**
```
{
  "error": "Build folder not found",
  "message": "Please run 'npm run build' first"
}
```

**Solution:**
```bash
cd talenthub-app
npm run build
```

**Why it happens:** The Flask server looks for compiled React files in the `build/` folder. If you haven't run `npm run build`, this folder doesn't exist.

---

### 2. Module Not Found Errors

**Error Message:**
```
Module not found: Can't resolve 'react'
```

**Solution:**
```bash
npm install
```

**Why it happens:** Node modules haven't been installed yet.

---

### 3. Python Module Errors

**Error Message:**
```
ModuleNotFoundError: No module named 'flask'
```

**Solution:**
```bash
pip install flask pymupdf pdfplumber python-docx pillow pytesseract spacy numpy --break-system-packages
python -m spacy download en_core_web_sm
```

**Why it happens:** Python dependencies are missing.

---

### 4. Port Already in Use

**Error Message:**
```
OSError: [Errno 48] Address already in use
```

**Solution Option 1:** Kill the process
```bash
# Find process
lsof -ti:5001

# Kill it
lsof -ti:5001 | xargs kill -9
```

**Solution Option 2:** Change the port

Edit `parse.py` (line ~4187):
```python
app.run(debug=False, port=5002, host='0.0.0.0')  # Changed to 5002
```

Also update `package.json`:
```json
"proxy": "http://localhost:5002"
```

---

### 5. Blank Page / White Screen

**Check 1: Browser Console**
Press F12 and check for errors like:
- 404 errors → build files not found
- CORS errors → backend not running

**Check 2: Build Files**
```bash
ls build/
# Should show: index.html, static/, etc.
```

**Check 3: Flask Running**
```bash
curl http://localhost:5001/api/stats
# Should return JSON data
```

**Solution:**
1. Ensure Flask is running: `python parse.py --web`
2. Rebuild React: `npm run build`
3. Clear browser cache (Ctrl+Shift+R)

---

### 6. API Calls Failing (404 Errors)

**Error in Browser Console:**
```
GET http://localhost:5001/api/candidates 404 (Not Found)
```

**Check:**
1. Is Flask running? Look for "Running on http://0.0.0.0:5001"
2. Check the route in `parse.py`:
   ```python
   @app.get('/api/candidates')
   def api_candidates():
   ```

**Solution:**
- Restart Flask: `python parse.py --web`
- Verify the endpoint exists in parse.py

---

### 7. Changes Not Appearing

**For Frontend Changes:**
1. Make changes to `.jsx` or `.css` files
2. Rebuild: `npm run build`
3. Refresh browser (Ctrl+Shift+R)

**For Backend Changes:**
1. Edit `parse.py`
2. Stop Flask (Ctrl+C)
3. Restart: `python parse.py --web`

---

### 8. npm run build Fails

**Error Message:**
```
Failed to compile
```

**Common Causes:**
1. **Syntax errors in JSX**
   - Check the error message for file and line number
   - Look for missing closing tags, brackets, etc.

2. **Import errors**
   ```javascript
   // Wrong
   import Analytics from './analytics'  // case sensitive!
   
   // Correct
   import Analytics from './Analytics'
   ```

3. **CSS syntax errors**
   - Check for missing semicolons or brackets in CSS files

**Solution:**
- Read the error message carefully
- Fix the indicated file
- Run `npm run build` again

---

### 9. Slow Performance / Hanging

**Symptoms:**
- Page takes forever to load
- API calls timeout

**Possible Causes:**
1. Large database (too many resumes)
2. Synchronous file operations
3. Memory issues

**Solutions:**
1. **Check database size:**
   ```bash
   du -h resumes_db.json
   ```
   If > 50MB, consider splitting or using a real database

2. **Check available memory:**
   ```bash
   free -h
   ```

3. **Limit results in components:**
   Edit component to fetch fewer items initially

---

### 10. File Upload Not Working

**Error Message:**
```
413 Payload Too Large
```

**Solution:**
Edit `parse.py` to increase limit:
```python
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
```

**Other Upload Issues:**
1. Check file permissions on `uploads/` folder
2. Verify file format is supported (PDF, DOCX, images)
3. Check Flask logs for parsing errors

---

### 11. Duplicates Not Being Removed

**Issue:** Clicking "Remove Duplicates" doesn't work

**Check:**
1. Flask terminal for errors
2. Browser console for failed API calls

**Solution:**
1. Verify endpoint exists in parse.py:
   ```python
   @app.post('/api/remove_duplicates')
   ```
2. Check database permissions
3. Restart Flask server

---

### 12. Candidate Status Not Updating

**Issue:** Dropdown changes but status doesn't persist

**Debug Steps:**
1. Open browser console
2. Change status
3. Look for API call: `POST /api/candidate/123/update`
4. Check response

**Common Causes:**
1. Backend route not implemented
2. Database write permissions
3. Candidate ID mismatch

**Solution:**
```javascript
// In Candidates.jsx, verify:
const updateCandidateStatus = async (candidateId, status) => {
  const response = await fetch(`${API_URL}/api/candidate/${candidateId}/update`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status })
  });
  // Check response
  console.log(await response.json());
};
```

---

## 🛠️ Debug Checklist

When something's not working:

- [ ] Flask server is running
- [ ] Browser console shows no errors (F12)
- [ ] `build/` folder exists
- [ ] API endpoints return data (test with curl)
- [ ] Node modules installed (`npm install`)
- [ ] Python packages installed
- [ ] Correct port numbers everywhere
- [ ] Files have correct permissions
- [ ] No syntax errors in code
- [ ] Cache cleared (hard refresh)

---

## 📞 Getting More Help

### Check Flask Logs
The terminal running `parse.py --web` shows all backend activity:
- API calls
- Errors
- File operations

### Check Browser Console
Press F12 in browser to see:
- JavaScript errors
- API call responses
- Network activity

### Test API Directly
```bash
# Test if backend is responding
curl http://localhost:5001/api/stats

# Test candidate endpoint
curl http://localhost:5001/api/candidates
```

### Verify File Structure
```bash
# Should show all necessary files
ls -la talenthub-app/
ls -la talenthub-app/src/
ls -la talenthub-app/build/
```

---

## 🔄 Clean Restart Process

If all else fails, do a complete restart:

```bash
# 1. Stop Flask (Ctrl+C)

# 2. Clean build
rm -rf build/
rm -rf node_modules/

# 3. Reinstall
npm install

# 4. Rebuild
npm run build

# 5. Start fresh
python parse.py --web
```

---

## 💡 Pro Tips

1. **Always check both terminals:** React dev server AND Flask backend
2. **Hard refresh in browser:** Ctrl+Shift+R (bypasses cache)
3. **Check file extensions:** `.jsx` not `.js` for React components
4. **Case matters:** `Analytics.jsx` ≠ `analytics.jsx`
5. **Watch for typos:** `Analytics.css` must match import exactly
6. **Read error messages:** They usually tell you exactly what's wrong
7. **One change at a time:** Easier to debug
8. **Use console.log:** Debug React state and API responses
9. **Check network tab:** See all API calls in browser dev tools
10. **Keep backups:** Before making major changes

---

Remember: Most issues are simple fixes like missing builds, wrong ports, or typos! 🎯
