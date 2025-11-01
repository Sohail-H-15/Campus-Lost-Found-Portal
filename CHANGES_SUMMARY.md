# Changes Summary - Campus Lost & Found Portal

## Overview
This document summarizes all the changes made to improve the functionality and user experience of the Lost & Found Portal application.

---

## 1. Image Search Filtering (60% Match Threshold)

### Backend Changes
**File**: `app.py` - `/match_images` endpoint
- No changes needed (filtering done on frontend)

### Frontend Changes
**File**: `index.html`

**Function Modified**: `performDedicatedImageSearch()` and `performImageSearch()`

**What Changed**:
```javascript
// Filter matches to only show those with similarity >= 60% (0.6)
const filteredMatches = data.matches.filter(match => match.similarity >= 0.6);
```

**How It Works**:
- After receiving image match results from backend, filters matches where `similarity >= 0.6`
- Displays only high-confidence matches (60% or above)
- Shows a message if no matches meet the threshold

**Key Technologies**:
- JavaScript Array `.filter()` method
- Similarity threshold comparison (0.6 = 60%)

---

## 2. Email Verification & Login Flow Improvements

### Frontend Changes
**File**: `index.html`

**Variables Added**:
```javascript
let signupEmail = null; // Store email from signup for pre-filling login
```

**Function Added**: `navigateToLogin()`
```javascript
function navigateToLogin() {
    // Navigate to auth page and show login form directly
    // Pre-fill email if available
    // Show login form, hide signup form
}
```

**Function Modified**: `signup()`
- Stores the email entered during signup: `signupEmail = data.email;`

**Function Modified**: `handleRouteChange()`
- Pre-fills resend verification email input when verification page is shown

**HTML Changes**:
- "Back to Login" button now calls `navigateToLogin()` instead of `navigateTo('auth')`
- Resend email input placeholder changed to "Enter your contact email"

**How It Works**:
1. User signs up → email is stored in `signupEmail` variable
2. User clicks "Back to Login" → `navigateToLogin()` function:
   - Shows login form directly (not signup)
   - Pre-fills email field with stored email
   - Pre-fills resend verification email input

**Key Technologies**:
- JavaScript DOM manipulation
- localStorage equivalent (in-memory variable)
- Event handling

---

## 3. Image Matching - Location & Date Display Fix

### Backend Changes
**File**: `app.py` - `/match_images` endpoint

**SQL Query Modified**:
```python
# Before:
"SELECT i.id as image_id, i.item_id, i.filename, i.feature_vector, "
"it.id, it.type, it.title, it.description, it.status "

# After:
"SELECT i.id as image_id, i.item_id, i.filename, i.feature_vector, "
"it.id, it.type, it.title, it.description, it.location, it.date_str, it.status "
```

**Response Data Modified**:
```python
'item': {
    'id': row['id'],
    'type': row['type'],
    'title': row['title'],
    'description': row['description'],
    'location': row['location'],      # Added
    'date_str': row['date_str'],      # Added
    'status': row['status'],
    'image_url': f"/uploads/{row['filename']}"
}
```

**How It Works**:
- SQL query now includes `it.location` and `it.date_str` fields
- These fields are included in the JSON response
- Frontend already had the display code, just needed the data

**Key Technologies**:
- SQL JOIN queries
- Python dictionary construction

---

## 4. My Items - Claim Priority Sorting

### Backend Changes
**File**: `app.py` - `/my_items` endpoint

**SQL Query Modified**:
```python
# Before:
"SELECT * FROM items WHERE user_id = ? ORDER BY created_at DESC"

# After:
"""
SELECT it.*, 
       CASE WHEN EXISTS (
           SELECT 1 FROM claims c 
           WHERE c.item_id = it.id AND c.status = 'pending'
       ) THEN 1 ELSE 0 END as has_claims
FROM items it 
WHERE it.user_id = ? 
ORDER BY has_claims DESC, it.created_at DESC
"""
```

**How It Works**:
1. Uses SQL `CASE WHEN EXISTS` to check for pending claims
2. Creates a virtual column `has_claims` (1 = has claims, 0 = no claims)
3. Orders by `has_claims DESC` first (items with claims come first)
4. Then orders by `created_at DESC` (newest first within each group)

**Key Technologies**:
- SQL subqueries
- SQL CASE statements
- SQL ORDER BY with multiple columns

---

## 5. Claim Status Management & Stats Update

### Backend Changes
**File**: `app.py`

**Modified**: `/stats` endpoint
```python
# Before:
"SELECT COUNT(*) as count FROM items WHERE type = 'found'"

# After:
"SELECT COUNT(*) as count FROM items WHERE type = 'found' AND status != 'returned'"
```

**Why**: When a found item is returned (claim approved), it should no longer count in "Items Found" statistics.

**Already Working**: `/resolve_claim` endpoint
- When claim is **approved**: Sets item status to `'returned'`
- When claim is **denied**: Sets item status back to `'open'` (if no other pending claims)

### Frontend Changes
**File**: `index.html`

**Function Modified**: `fetchMyItems()`

**Status Display Logic**:
```javascript
// Display status: if pending and has claims, show "claim pending"
let statusDisplay = item.status;
if (item.status === 'pending' && item.claims && item.claims.length > 0) {
    statusDisplay = 'claim pending';
}

// Status badge colors
let statusColor = 'bg-blue-200 text-blue-800';
if (statusDisplay === 'claim pending') {
    statusColor = 'bg-yellow-200 text-yellow-800';
} else if (statusDisplay === 'returned') {
    statusColor = 'bg-green-200 text-green-800';
} else if (statusDisplay === 'open') {
    statusColor = 'bg-blue-200 text-blue-800';
}
```

**How It Works**:
- Checks if item status is 'pending' AND has claims
- Displays "claim pending" with yellow badge instead of "pending"
- Different colors for different statuses (yellow for pending claims, green for returned, blue for open)

**Key Technologies**:
- JavaScript conditional logic
- CSS classes for styling (Tailwind CSS)
- Template string interpolation

---

## 6. Notification Badges for Pending Claims

### Frontend Changes
**File**: `index.html`

**HTML Elements Added**:
1. Header badge: `<span id="myItemsNotificationBadge">` on "My Items" button
2. Home page badge: `<span id="myItemsHomeNotificationBadge">` on "My Posts & Claims" card

**Function Added**: `updateMyItemsNotification(count)`
```javascript
function updateMyItemsNotification(count) {
    const badge = document.getElementById('myItemsNotificationBadge');
    const homeBadge = document.getElementById('myItemsHomeNotificationBadge');
    
    if (count > 0) {
        // Show badges with count
        badge.textContent = count;
        badge.classList.remove('hidden');
        // Same for homeBadge
    } else {
        // Hide badges
        badge.classList.add('hidden');
    }
}
```

**Function Modified**: `fetchMyItems()`
```javascript
// Count items with pending claims
let pendingClaimsCount = 0;
if (data.items && data.items.length > 0) {
    pendingClaimsCount = data.items.filter(item => 
        item.claims && item.claims.length > 0
    ).length;
}

// Update notification badges
updateMyItemsNotification(pendingClaimsCount);
```

**Function Modified**: `handleRouteChange()`
- Updates notification count when navigating to home page

**How It Works**:
1. When fetching items, counts how many have pending claims
2. Calls `updateMyItemsNotification()` to update both badges
3. Badges show red circular indicators with the count
4. Badges are hidden when count is 0
5. Updates automatically when navigating or resolving claims

**Key Technologies**:
- JavaScript DOM manipulation
- Array `.filter()` method
- CSS for badge styling (position: absolute, rounded-full)
- Event-driven updates

---

## Technical Stack Summary

### Backend (Python/Flask)
- **Framework**: Flask
- **Database**: SQLite with sqlite3
- **Image Processing**: 
  - ResNet50 (PyTorch/torchvision) or SIFT (OpenCV)
  - Feature extraction and similarity comparison
- **Email**: Flask-Mail with SMTP

### Frontend (JavaScript/HTML)
- **Styling**: Tailwind CSS
- **JavaScript**: Vanilla JavaScript (no frameworks)
- **Features Used**:
  - Fetch API for HTTP requests
  - DOM manipulation
  - Event handling
  - Template literals
  - Array methods (filter, forEach, map)

### Database Schema
**Tables Used**:
- `users`: User accounts with email verification
- `items`: Lost/found items (with location, date_str, status)
- `images`: Image files with feature vectors
- `claims`: Claim requests (status: pending/approved/denied)

---

## Data Flow Examples

### 1. Image Search Flow
```
User uploads image
  ↓
Frontend: performDedicatedImageSearch()
  ↓
POST /match_images with image file
  ↓
Backend: Extract features → Compare with database → Return matches
  ↓
Frontend: Filter matches >= 60% similarity → Display results
```

### 2. Claim Notification Flow
```
User claims item
  ↓
POST /claim_item
  ↓
Backend: Set item status to 'pending' → Create claim record
  ↓
User (owner) navigates to "My Items"
  ↓
GET /my_items
  ↓
Backend: Query items with claims first → Return with claims array
  ↓
Frontend: Count items with claims → Update notification badges
```

### 3. Claim Resolution Flow
```
Owner approves/denies claim
  ↓
POST /resolve_claim
  ↓
Backend: 
  - If approved: Set item status to 'returned'
  - If denied: Set item status to 'open'
  ↓
Frontend: Refresh items list → Update notification badges
  ↓
Stats update: Found items count decreases if approved
```

---

## Key Design Patterns Used

1. **RESTful API**: Backend uses REST endpoints
2. **Single Page Application (SPA)**: Frontend handles routing client-side
3. **Database Normalization**: Separate tables for users, items, images, claims
4. **Feature Extraction**: AI-based image matching using deep learning/computer vision
5. **Status Management**: State machine pattern for item statuses (open → pending → returned)

---

## Performance Considerations

1. **Image Matching**: Uses pre-computed feature vectors stored in database (BLOB)
2. **Sorting**: Database-level sorting (more efficient than sorting in application)
3. **Notification Updates**: Only updates when necessary (navigation or claim resolution)
4. **Filtering**: Client-side filtering for match threshold (reduces network overhead)

---

## Future Enhancement Opportunities

1. Real-time notifications (WebSocket integration)
2. Email notifications for claim updates
3. Pagination for large result sets
4. Caching for frequently accessed items
5. Image compression before storage
6. Advanced search filters (date range, location, etc.)

---

## Testing Checklist

- [x] Image search shows only 60%+ matches
- [x] "Back to Login" shows login form with pre-filled email
- [x] Image matches show location and date
- [x] Items with claims appear first in "My Items"
- [x] Status shows "claim pending" when item has claims
- [x] Denied claims set item back to "open"
- [x] Approved claims decrease found items count
- [x] Notification badges appear when items have pending claims

