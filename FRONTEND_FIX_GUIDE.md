# Fix for FormData.append() Error

## Problem
The error `TypeError: Failed to execute 'append' on 'FormData': parameter 2 is not of type 'Blob'` occurs when trying to append non-Blob data to FormData.

## Solution

In your `VisualSearch.ts` file, ensure you're converting your sketch/image data to a Blob before appending to FormData.

### Common Scenarios:

#### 1. If you have a Canvas Element
```typescript
// VisualSearch.ts
export async function searchBySketch(canvas: HTMLCanvasElement, query?: string) {
    const formData = new FormData();
    
    // Convert canvas to Blob
    return new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) {
                formData.append('file', blob, 'sketch.png'); // ✅ Blob is correct type
                if (query) {
                    formData.append('query', query);
                }
                resolve(blob);
            } else {
                reject(new Error('Failed to convert canvas to blob'));
            }
        }, 'image/png');
    }).then(() => {
        // Make the API call
        return fetch('http://your-backend-url/search', {
            method: 'POST',
            body: formData
        });
    });
}
```

#### 2. If you have a Base64 String
```typescript
// Convert base64 to Blob
function base64ToBlob(base64: string, mimeType: string = 'image/png'): Blob {
    const byteCharacters = atob(base64.split(',')[1] || base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

export async function searchBySketch(base64Image: string, query?: string) {
    const formData = new FormData();
    const blob = base64ToBlob(base64Image, 'image/png'); // ✅ Convert to Blob
    formData.append('file', blob, 'sketch.png');
    if (query) {
        formData.append('query', query);
    }
    
    return fetch('http://your-backend-url/search', {
        method: 'POST',
        body: formData
    });
}
```

#### 3. If you have a Data URL (data:image/...)
```typescript
// Convert data URL to Blob
function dataURLToBlob(dataURL: string): Blob {
    const arr = dataURL.split(',');
    const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/png';
    const bstr = atob(arr[1] || arr[0]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
}

export async function searchBySketch(dataURL: string, query?: string) {
    const formData = new FormData();
    const blob = dataURLToBlob(dataURL); // ✅ Convert to Blob
    formData.append('file', blob, 'sketch.png');
    if (query) {
        formData.append('query', query);
    }
    
    return fetch('http://your-backend-url/search', {
        method: 'POST',
        body: formData
    });
}
```

#### 4. Complete Example with Error Handling
```typescript
// VisualSearch.ts
export async function searchBySketch(
    canvasOrImage: HTMLCanvasElement | string, 
    query?: string
): Promise<Response> {
    const formData = new FormData();
    let blob: Blob;
    
    try {
        // Handle different input types
        if (canvasOrImage instanceof HTMLCanvasElement) {
            // Canvas element - convert to Blob
            blob = await new Promise<Blob>((resolve, reject) => {
                canvasOrImage.toBlob((blob) => {
                    if (blob) {
                        resolve(blob);
                    } else {
                        reject(new Error('Failed to convert canvas to blob'));
                    }
                }, 'image/png');
            });
        } else if (typeof canvasOrImage === 'string') {
            // String - could be base64 or data URL
            if (canvasOrImage.startsWith('data:')) {
                // Data URL
                const arr = canvasOrImage.split(',');
                const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/png';
                const bstr = atob(arr[1]);
                const u8arr = new Uint8Array(bstr.length);
                for (let i = 0; i < bstr.length; i++) {
                    u8arr[i] = bstr.charCodeAt(i);
                }
                blob = new Blob([u8arr], { type: mime });
            } else {
                // Base64 string
                const byteCharacters = atob(canvasOrImage);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                blob = new Blob([new Uint8Array(byteNumbers)], { type: 'image/png' });
            }
        } else {
            throw new Error('Invalid input type. Expected HTMLCanvasElement or string.');
        }
        
        // ✅ Now append the Blob to FormData
        formData.append('file', blob, 'sketch.png');
        
        if (query) {
            formData.append('query', query);
        }
        
        // Make the API call
        const response = await fetch('http://your-backend-url/search', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }
        
        return response;
    } catch (error) {
        console.error('Error in searchBySketch:', error);
        throw error;
    }
}
```

## Key Points:
1. **Always convert to Blob** before appending to FormData
2. **Check the type** of your input data (canvas, base64, data URL, etc.)
3. **Use the correct conversion method** for each data type
4. **Handle errors** appropriately

## Backend is Ready
The backend now accepts:
- POST request to `/search`
- FormData with:
  - `file`: Blob/File (required)
  - `query`: string (optional)

Make sure to update the fetch URL in your frontend code to point to your backend server!
