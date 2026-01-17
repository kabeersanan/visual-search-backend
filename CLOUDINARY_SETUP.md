# Cloudinary Integration Guide

The backend now uses Cloudinary for cloud-based image storage. All images are uploaded to Cloudinary and Cloudinary URLs are returned instead of local file paths.

## Setup

### 1. Install Dependencies

Make sure `python-dotenv` and `cloudinary` are installed:

```bash
pip install cloudinary python-dotenv
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Get Cloudinary Credentials

1. Sign up for a free account at [Cloudinary](https://cloudinary.com/)
2. Go to your [Dashboard](https://cloudinary.com/console)
3. Copy your credentials:
   - Cloud Name
   - API Key
   - API Secret

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Cloudinary credentials:

```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

**Important:** Never commit your `.env` file to git! It's already in `.gitignore`.

## How It Works

### `/store` Endpoint

When you upload an image via `/store`:

1. **File Validation**: Validates file type and size (max 10MB)
2. **Upload to Cloudinary**: Uploads image to Cloudinary in the `brandkit` folder
3. **Generate Embedding**: Creates CLIP embedding from the uploaded image
4. **Store Metadata**: Saves embedding + Cloudinary URL to `image_embeddings.pkl`
5. **Return URL**: Returns Cloudinary URL in the response

**Response Format:**
```json
{
  "message": "Brandkit image stored successfully",
  "filename": "logo_20250116_123456_abc123.png",
  "cloudinary_url": "https://res.cloudinary.com/your_cloud/image/upload/v1234567890/brandkit/logo_20250116_123456_abc123.png",
  "embedding_generated": true,
  "total_images_in_database": 5,
  "metadata": {
    "filename": "...",
    "original_filename": "...",
    "content_type": "image/png",
    "size": 24567,
    "uploaded_at": "2025-01-16T12:34:56.789Z",
    "embedding_generated": true
  }
}
```

### `/search` Endpoint

When you search for similar images:

1. **Temporary Upload**: Uploads search query image temporarily (for embedding generation)
2. **Generate Embedding**: Creates CLIP embedding from query image
3. **Find Similar**: Compares with stored embeddings using cosine similarity
4. **Return URLs**: Returns similar images with their Cloudinary URLs

**Response Format:**
```json
{
  "message": "Search completed successfully",
  "query": null,
  "query_image": {
    "original_filename": "search.png",
    "saved_filename": "20250116_123456_abc123.png",
    "content_type": "image/png",
    "size": 12345
  },
  "similar_images": [
    {
      "filename": "logo_20250116_123456_abc123.png",
      "similarity": 0.95,
      "cloudinary_url": "https://res.cloudinary.com/your_cloud/image/upload/v1234567890/brandkit/logo_20250116_123456_abc123.png"
    },
    {
      "filename": "icon_20250115_234567_def456.png",
      "similarity": 0.87,
      "cloudinary_url": "https://res.cloudinary.com/your_cloud/image/upload/v1234567890/brandkit/icon_20250115_234567_def456.png"
    }
  ],
  "total_results": 2,
  "uploaded_at": "2025-01-16T12:34:56.789Z"
}
```

## Frontend Usage

### Upload Image to Brandkit

```typescript
const formData = new FormData();
formData.append('file', fileObject);

const response = await fetch('https://localhost:8443/store', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.cloudinary_url); // Use this URL directly in <img> tags

// Display image
<img src={result.cloudinary_url} alt="Brandkit image" />
```

### Search for Similar Images

```typescript
const formData = new FormData();
formData.append('file', searchImage);
formData.append('top_k', '5');

const response = await fetch('https://localhost:8443/search', {
  method: 'POST',
  body: formData
});

const result = await response.json();

// Display similar images using Cloudinary URLs
result.similar_images.forEach(img => {
  <img src={img.cloudinary_url} alt={img.filename} />
});
```

## Image Organization

All images are stored in Cloudinary under the `brandkit` folder:
- Path: `brandkit/{filename}`
- Example: `brandkit/logo_20250116_123456_abc123.png`

## Data Migration

The code automatically migrates old embeddings (without Cloudinary URLs) to the new format:
- Old format: `{filename: embedding_array}`
- New format: `{filename: {"embedding": [...], "cloudinary_url": "https://..."}}`

If you have existing embeddings without Cloudinary URLs, they will work but won't have URLs until you re-upload the images.

## Troubleshooting

### Error: "Failed to upload to Cloudinary"

**Possible causes:**
1. Missing or incorrect environment variables
2. Invalid Cloudinary credentials
3. Network connectivity issues

**Solution:**
- Check your `.env` file has correct credentials
- Verify credentials in Cloudinary dashboard
- Check network connection

### Images Not Showing

**Possible causes:**
1. Cloudinary URL is `null` in response
2. Image upload failed but embedding was generated

**Solution:**
- Check the response `cloudinary_url` field
- Verify Cloudinary upload was successful (check Cloudinary dashboard)
- Re-upload the image if needed

### Embedding Generation Fails

**Possible causes:**
1. CLIP model not loaded
2. Invalid image format
3. Image processing error

**Solution:**
- Check backend logs for error messages
- Verify image format is supported (PNG, JPG, JPEG, WEBP)
- Ensure CLIP model loaded successfully on startup

## Benefits of Cloudinary

✅ **Scalable**: No local storage limits
✅ **CDN**: Fast global delivery via Cloudinary CDN
✅ **Transformations**: Built-in image transformations (resize, crop, etc.)
✅ **Reliability**: High availability and backup
✅ **Optimization**: Automatic image optimization

## Cloudinary URL Transformations

You can modify Cloudinary URLs for transformations:

```typescript
// Original URL
const url = "https://res.cloudinary.com/cloud/image/upload/v123/brandkit/logo.png";

// Resize to 300x300
const resized = url.replace("/upload/", "/upload/w_300,h_300,c_fill/");

// Apply quality optimization
const optimized = url.replace("/upload/", "/upload/q_auto,f_auto/");
```

See [Cloudinary Transformations](https://cloudinary.com/documentation/image_transformations) for more options.
