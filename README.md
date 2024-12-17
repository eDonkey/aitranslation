Here's a comprehensive README.md for the API functionalities:

```markdown:README.md
# Translation Service API Documentation

This API provides translation services with both AI-powered and human translation capabilities. It includes authentication, text management, and administrative functions.

## Authentication

All endpoints require authentication using an API key passed in the header:

```bash
X-API-Key: your-api-key
```

For admin endpoints, use the admin key:
```bash
X-Admin-Key: your-admin-key
```

## Endpoints

### Translation Endpoints

#### AI Translation
```bash
POST /translate-text/
```
Translates text to all available languages using AI.

**Request:**
```json
{
    "text": "Hello world",
    "language": "english"
}
```
Supported languages: "english", "spanish", "portuguese", "french", "deutch", "italian"

**Response:**
```json
{
    "original_text": "Hello world",
    "original_language": "english",
    "translations": {
        "spanish": "Hola mundo",
        "portuguese": "Olá mundo",
        "french": "Bonjour le monde",
        "deutch": "Hallo Welt",
        "italian": "Ciao mondo"
    },
    "id": 1
}
```

#### Save Text for Human Translation
```bash
POST /save-text/
```
Submits text for human translation.

**Request:**
```json
{
    "text": "Hello world",
    "language": "english"
}
```

**Response:**
```json
{
    "id": 1,
    "message": "Text saved successfully for translation"
}
```

### Text Retrieval Endpoints

#### Get All Texts
```bash
GET /get-texts/
```
Retrieves all texts associated with the API key.

**Response:**
```json
[
    {
        "id": 1,
        "english": "Hello world",
        "spanish": "Hola mundo",
        "portuguese": "Olá mundo",
        "french": "Bonjour le monde",
        "deutch": "Hallo Welt",
        "italian": "Ciao mondo",
        "created_at": "2024-01-20T15:30:00Z",
        "updated_at": "2024-01-20T15:30:00Z"
    }
]
```

#### Get Specific Text
```bash
GET /get-text/{text_id}
```
Retrieves a specific text by ID.

**Response:**
```json
{
    "id": 1,
    "translations": {
        "english": "Hello world",
        "spanish": "Hola mundo"
    },
    "metadata": {
        "created_at": "2024-01-20T15:30:00Z",
        "updated_at": "2024-01-20T15:30:00Z",
        "api_key_name": "Client A"
    }
}
```

### Admin Endpoints

#### List API Keys
```bash
GET /admin/api-keys/
```
Lists all API keys.

**Response:**
```json
[
    {
        "id": 1,
        "key": "api-key-value",
        "name": "Client A",
        "created_at": "2024-01-20T15:30:00Z",
        "last_used": "2024-01-20T16:45:00Z"
    }
]
```

#### Create API Key
```bash
POST /admin/api-keys/
```
Creates a new API key.

**Request:**
```json
{
    "name": "New Client"
}
```

**Response:**
```json
{
    "id": 2,
    "key": "new-api-key-value",
    "name": "New Client",
    "created_at": "2024-01-20T17:00:00Z"
}
```

#### Delete API Key
```bash
DELETE /admin/api-keys/{key_id}
```
Deletes an API key.

#### Get Pending Translations
```bash
GET /admin/pending-translations
```
Lists texts pending human translation.

**Response:**
```json
[
    {
        "id": 1,
        "original_language": "english",
        "text": "Hello world",
        "missing_translations": ["spanish", "french"],
        "created_at": "2024-01-20T15:30:00Z",
        "api_key_name": "Client A"
    }
]
```

#### Save Human Translation
```bash
POST /admin/translations/{entry_id}
```
Saves a human translation for a specific text.

**Request:**
```json
{
    "language": "spanish",
    "text": "Hola mundo"
}
```

**Response:**
```json
{
    "message": "Translation saved successfully"
}
```

## Error Responses

The API uses standard HTTP status codes:

- 200: Success
- 400: Bad Request
- 401: Unauthorized (Invalid API key)
- 403: Forbidden (Insufficient permissions)
- 404: Not Found
- 422: Validation Error
- 500: Server Error

Error response format:
```json
{
    "detail": "Error message here"
}
```

## Rate Limiting

- AI translations: 100 requests per day per API key
- Human translations: Unlimited submissions
- API key creation: Admin only
- Text retrieval: Unlimited requests

## Notes

- All timestamps are in UTC
- Text length is limited to 5000 characters
- API keys should be kept secure and not shared
- Admin key has access to all endpoints
- Regular API keys can only access their own texts
```

This README provides:
1. Authentication details
2. Complete endpoint documentation
3. Request/response examples
4. Error handling information
5. Rate limiting details
6. Important notes and limitations