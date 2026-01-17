# Image Blob Display Guide

The backend now supports returning images as blobs/base64 for direct display in the frontend.

## Two Ways to Display Images

### Option 1: Use Image URLs (Recommended for Large Images)

The backend provides image URLs that you can use directly in `<img>` tags:

**Response includes:**
```json
{
  "similar_images": [
    {
      "filename": "logo.png",
      "similarity": 0.95,
      "image_url": "/images/logo.png"
    }
  ]
}
```

**Frontend Usage:**
```typescript
// Display image using URL
<img src={`https://localhost:8443${image.image_url}`} alt={image.filename} />

// Or in React/JSX
{similarImages.map(img => (
  <img key={img.filename} src={`https://localhost:8443${img.image_url}`} />
))}
```

### Option 2: Use Base64 Encoded Images (Good for Small Images)

Enable base64 encoding by setting `include_image_data=true` in your request:

**Request:**
```typescript
const formData = new FormData();
formData.append('file', file);
formData.append('include_image_data', 'true'); // Enable base64

const response = await fetch('/search', {
  method: 'POST',
  body: formData
});
```

**Response includes:**
```json
{
  "similar_images": [
    {
      "filename": "logo.png",
      "similarity": 0.95,
      "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

**Frontend Usage:**
```typescript
// Direct display - no additional fetch needed
<img src={image.image_data} alt={image.filename} />

// Or in React/JSX
{similarImages.map(img => (
  <img key={img.filename} src={img.image_data} />
))}
```

## Endpoints

### 1. GET `/images/{filename}` - Serve Image as Blob

Returns the raw image file with proper Content-Type headers.

```typescript
// Direct usage in img tag
<img src="https://localhost:8443/images/logo.png" />

// Or fetch and create blob URL
const response = await fetch('https://localhost:8443/images/logo.png');
const blob = await response.blob();
const blobUrl = URL.createObjectURL(blob);
// Use blobUrl in img src
```

### 2. POST `/store` - Store Image (Returns Base64)

The `/store` endpoint automatically returns the uploaded image as base64:

**Response:**
```json
{
  "message": "Brandkit image stored successfully",
  "filename": "logo.png",
  "image_url": "/images/logo.png",
  "image_data": "data:image/png;base64,...",  // ✅ Ready to display!
  "embedding_generated": true
}
```

**Frontend Usage:**
```typescript
const formData = new FormData();
formData.append('file', file);

const response = await fetch('/store', {
  method: 'POST',
  body: formData
});

const result = await response.json();
// Display immediately
<img src={result.image_data} alt="Uploaded image" />
```

### 3. POST `/search` - Search with Optional Image Data

**Request with base64 (for immediate display):**
```typescript
const formData = new FormData();
formData.append('file', searchImage);
formData.append('include_image_data', 'true'); // Enable base64
formData.append('top_k', '5');

const response = await fetch('/search', {
  method: 'POST',
  body: formData
});

const result = await response.json();
// Display similar images immediately
result.similar_images.forEach(img => {
  console.log(img.image_data); // Base64 data URL
});
```

**Request without base64 (lighter, use URLs):**
```typescript
const formData = new FormData();
formData.append('file', searchImage);
// Don't set include_image_data, or set to 'false'
formData.append('top_k', '5');

const response = await fetch('/search', {
  method: 'POST',
  body: formData
});

const result = await response.json();
// Use image URLs
result.similar_images.forEach(img => {
  const imageUrl = `https://localhost:8443${img.image_url}`;
  console.log(imageUrl);
});
```

## Performance Considerations

### When to Use Base64 (`include_image_data=true`):
- ✅ Small images (< 100KB)
- ✅ Need immediate display without additional requests
- ✅ Few results (< 5 images)
- ✅ Offline capability needed

### When to Use Image URLs (`include_image_data=false`):
- ✅ Large images (> 100KB)
- ✅ Many results (10+ images)
- ✅ Better performance (lazy loading)
- ✅ Browser caching benefits
- ✅ Lower memory usage

## Complete Example

```typescript
// Store an image
async function storeBrandkit(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('https://localhost:8443/store', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  
  // Display immediately using base64
  const img = document.createElement('img');
  img.src = result.image_data; // Base64 data URL
  document.body.appendChild(img);
}

// Search with base64 images
async function searchWithImages(searchFile: File) {
  const formData = new FormData();
  formData.append('file', searchFile);
  formData.append('include_image_data', 'true');
  formData.append('top_k', '5');
  
  const response = await fetch('https://localhost:8443/search', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  
  // Display all similar images
  result.similar_images.forEach(img => {
    const imgElement = document.createElement('img');
    imgElement.src = img.image_data; // Base64 data URL
    imgElement.alt = img.filename;
    document.body.appendChild(imgElement);
  });
}
```

## Security Note

The `/images/{filename}` endpoint includes security checks:
- Prevents path traversal attacks (`..`, `/`, `\`)
- Only serves files from the `photos/` directory
- Validates file extensions
