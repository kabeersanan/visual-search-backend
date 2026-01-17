# Store API Endpoint - Frontend Integration Guide

## Endpoint: POST /store

## The Error: "Unprocessable Entity" (422)

This error occurs when the request format doesn't match what FastAPI expects. The `/store` endpoint requires **FormData** with a specific format.

## Correct Frontend Implementation

### ✅ Correct Way (TypeScript/JavaScript)

```typescript
async function storeBrandkitImage(file: File, metadata?: Record<string, any>) {
    const formData = new FormData();
    
    // ✅ IMPORTANT: Append the File object directly (not JSON stringified)
    formData.append('file', file);
    
    // Optional: Add metadata as JSON string
    if (metadata) {
        formData.append('metadata', JSON.stringify(metadata));
    }
    
    try {
        const response = await fetch('https://localhost:8443/store', {
            method: 'POST',
            body: formData
            // ❌ DO NOT set Content-Type header - browser will set it automatically with boundary
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to store image');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error storing image:', error);
        throw error;
    }
}

// Usage:
const fileInput = document.querySelector('input[type="file"]');
const file = fileInput.files[0];

storeBrandkitImage(file, { category: 'logo', brand: 'MyBrand' })
    .then(result => console.log('Success:', result))
    .catch(error => console.error('Error:', error));
```

### ✅ Alternative: Using Blob from Canvas

```typescript
async function storeBrandkitFromCanvas(canvas: HTMLCanvasElement) {
    const formData = new FormData();
    
    // Convert canvas to Blob
    canvas.toBlob((blob) => {
        if (!blob) {
            throw new Error('Failed to convert canvas to blob');
        }
        
        // Create a File object from Blob
        const file = new File([blob], 'brandkit.png', { type: 'image/png' });
        formData.append('file', file);
        
        fetch('https://localhost:8443/store', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => console.log('Stored:', data))
        .catch(error => console.error('Error:', error));
    }, 'image/png');
}
```

## ❌ Common Mistakes

### 1. Sending as JSON (WRONG)
```typescript
// ❌ DON'T DO THIS
const response = await fetch('/store', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ file: fileObject })  // ❌ This won't work!
});
```

### 2. Setting Content-Type manually (WRONG)
```typescript
// ❌ DON'T DO THIS
const formData = new FormData();
formData.append('file', file);

await fetch('/store', {
    method: 'POST',
    headers: {
        'Content-Type': 'multipart/form-data'  // ❌ Browser sets this automatically!
    },
    body: formData
});
```

### 3. Stringifying the File object (WRONG)
```typescript
// ❌ DON'T DO THIS
formData.append('file', JSON.stringify(file));  // ❌ File must be appended directly
```

## React Example

```tsx
import { useState } from 'react';

function BrandkitUpload() {
    const [file, setFile] = useState<File | null>(null);
    
    const handleUpload = async () => {
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Optional metadata
        formData.append('metadata', JSON.stringify({
            category: 'logo',
            brand: 'MyBrand'
        }));
        
        try {
            const response = await fetch('https://localhost:8443/store', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }
            
            const result = await response.json();
            console.log('Upload successful:', result);
        } catch (error) {
            console.error('Upload error:', error);
        }
    };
    
    return (
        <div>
            <input 
                type="file" 
                accept="image/*" 
                onChange={(e) => setFile(e.target.files?.[0] || null)} 
            />
            <button onClick={handleUpload}>Upload Brandkit</button>
        </div>
    );
}
```

## Testing with cURL

```bash
curl -X POST "https://localhost:8443/store" \
  -F "file=@Screenshot 2026-01-16 233344.png" \
  -F 'metadata={"category":"logo"}' \
  -k
```

## Expected Response

```json
{
  "message": "Brandkit image stored successfully",
  "filename": "Screenshot_2026-01-16_233344_20260116_233344_a1b2c3d4.png",
  "saved_path": "photos/Screenshot_2026-01-16_233344_20260116_233344_a1b2c3d4.png",
  "metadata": {
    "filename": "Screenshot_2026-01-16_233344_20260116_233344_a1b2c3d4.png",
    "original_filename": "Screenshot 2026-01-16 233344.png",
    "filepath": "photos/Screenshot_2026-01-16_233344_20260116_233344_a1b2c3d4.png",
    "content_type": "image/png",
    "size": 24953,
    "uploaded_at": "2026-01-16T23:33:44.123456"
  }
}
```

## Troubleshooting

1. **Check browser console** for the actual error details
2. **Verify FormData** is being created correctly
3. **Ensure file is a File/Blob object**, not a string or base64
4. **Don't set Content-Type header** - let the browser set it automatically
5. **Check CORS** - make sure your origin is allowed
