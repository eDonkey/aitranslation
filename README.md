# AI Translation Service API Documentation

A comprehensive translation API service that provides AI-powered translations using OpenAI's GPT models. This service supports both text and CSV file translations across multiple languages with authentication and usage tracking.

## Table of Contents

- [Authentication](#authentication)
- [Supported Languages](#supported-languages)
- [Translation Endpoints](#translation-endpoints)
- [User Management (Admin)](#user-management-admin)
- [Error Handling](#error-handling)
- [Implementation Examples](#implementation-examples)
- [Rate Limiting & Costs](#rate-limiting--costs)

## Authentication

All API endpoints require authentication using an API key in the request headers:

```http
X-API-Key: your-api-key-here
```

### API Key Types

1. **User API Key**: For translation endpoints and user operations
2. **Master API Key**: For administrative operations (user management, statistics)

## Supported Languages

The API supports translation between the following languages:

| Language | Code |
|----------|------|
| English | `english` |
| Spanish | `spanish` |
| Portuguese | `portuguese` |
| French | `french` |
| German | `deutch` |
| Italian | `italian` |
| Filipino | `filipino` |
| Japanese | `japanese` |
| Vietnamese | `vietnamese` |

## Translation Endpoints

### Text Translation

**Endpoint:** `POST /translate-text/`

Translates a single text from a source language to one or more target languages.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to translate (max 5000 characters) |
| `source_language` | string | Yes | Source language code |
| `target_languages` | array | No | Target languages (if empty, translates to all except source) |
| `translation_style` | string | No | Translation style (e.g., "formal", "casual", "technical") |

#### Request Example

```http
POST /translate-text/
Content-Type: multipart/form-data
X-API-Key: your-api-key

text=Hello world, how are you?
source_language=english
target_languages=spanish
target_languages=french
translation_style=formal
```

#### Response Format

```json
{
    "message": "Text translated successfully",
    "id": 123,
    "original_text": "Hello world, how are you?",
    "original_language": "english",
    "translations": {
        "spanish": "Hola mundo, ¿cómo estás?",
        "french": "Bonjour le monde, comment allez-vous?"
    }
}
```

### CSV File Translation

**Endpoint:** `POST /translate-csv/`

Translates text content from a CSV file in batch mode.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | CSV file with text in first column |
| `source_language` | string | Yes | Source language code |
| `target_languages` | array | No | Target languages |
| `translation_style` | string | No | Translation style |

#### CSV File Format

Your input CSV should have the text to translate in the first column:

```csv
Text to translate
Hello world
How are you today?
Welcome to our service
```

#### Response

Returns a CSV file with original text and translations:

```csv
Original,spanish,french
Hello world,Hola mundo,Bonjour le monde
How are you today?,¿Cómo estás hoy?,Comment allez-vous aujourd'hui?
Welcome to our service,Bienvenido a nuestro servicio,Bienvenue dans notre service
```

### Get Translation by ID

**Endpoint:** `GET /api/translation/{translation_id}`

Retrieves a specific translation by its ID.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `translation_id` | integer | Yes | Translation ID |
| `translation_type` | string | No | Type filter: "text" or "csv" (default: "text") |

#### Response Format

```json
{
    "id": 123,
    "type": "text",
    "original_text": "Hello world",
    "source_language": "english",
    "translations": {
        "spanish": "Hola mundo",
        "french": "Bonjour le monde"
    },
    "style": "formal",
    "created_at": "2024-01-20T15:30:00Z",
    "model_version": "gpt-4-turbo"
}
```

### Download CSV Translation

**Endpoint:** `GET /api/translation/{translation_id}/download`

Downloads the translated CSV file for a specific translation.

#### Response

Returns the CSV file as a download with `Content-Disposition: attachment`.

### List User Translations

**Endpoint:** `GET /api/translations/`

Lists all translations for the authenticated user.

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `translation_type` | string | "all" | Filter by type: "all", "text", "csv" |
| `limit` | integer | 50 | Maximum results (max 100) |
| `offset` | integer | 0 | Results offset for pagination |

#### Response Format

```json
{
    "total": 245,
    "translations": [
        {
            "id": 123,
            "type": "text",
            "source_language": "english",
            "target_languages": ["spanish", "french"],
            "created_at": "2024-01-20T15:30:00Z",
            "cost": 0.02
        }
    ]
}
```

## User Management (Admin)

### Create User

**Endpoint:** `POST /api/admin/users/`

Creates a new user account with API key.

#### Request Body

```json
{
    "username": "john_doe",
    "email": "john@example.com",
    "is_active": true
}
```

#### Response

```json
{
    "id": 456,
    "username": "john_doe",
    "email": "john@example.com",
    "is_active": true,
    "api_key": "generated-api-key-here",
    "created_at": "2024-01-20T15:30:00Z"
}
```

### List Users

**Endpoint:** `GET /api/admin/users/`

Lists all users in the system.

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Maximum results |
| `offset` | integer | 0 | Results offset |

### Get User Statistics

**Endpoint:** `GET /api/admin/users/{user_id}/statistics`

Gets detailed usage statistics for a specific user.

#### Response

```json
{
    "user_id": 456,
    "period": "monthly",
    "totals": {
        "total_text_translations": 45,
        "total_csv_translations": 5,
        "total_api_calls": 50,
        "total_openai_cost": 12.45,
        "total_characters_translated": 125000
    },
    "language_breakdown": {
        "spanish": 20,
        "french": 15,
        "german": 10
    },
    "csv_metrics": {
        "rows_processed": 500,
        "avg_cost_per_csv": 2.49
    }
}
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Invalid data format |
| 500 | Internal Server Error |

### Error Response Format

```json
{
    "detail": "Error description here"
}
```

### Common Error Scenarios

1. **Invalid API Key**
```json
{
    "detail": "Invalid API Key"
}
```

2. **Invalid Language**
```json
{
    "detail": "Invalid target language: chinese. Allowed languages are: ['english', 'spanish', 'portuguese', 'french', 'deutch', 'italian', 'filipino', 'japanese', 'vietnamese']"
}
```

3. **Translation Failed**
```json
{
    "detail": "Translation failed: OpenAI API error"
}
```

## Implementation Examples

### Python

```python
import requests

# Configuration
BASE_URL = "https://your-api-domain.com"
API_KEY = "your-api-key-here"

headers = {
    "X-API-Key": API_KEY
}

# Text Translation
def translate_text(text, source_lang, target_langs=None, style=None):
    url = f"{BASE_URL}/translate-text/"
    
    data = {
        "text": text,
        "source_language": source_lang
    }
    
    if target_langs:
        data["target_languages"] = target_langs
    
    if style:
        data["translation_style"] = style
    
    response = requests.post(url, headers=headers, data=data)
    return response.json()

# CSV Translation
def translate_csv(file_path, source_lang, target_langs=None):
    url = f"{BASE_URL}/translate-csv/"
    
    data = {
        "source_language": source_lang
    }
    
    if target_langs:
        data["target_languages"] = target_langs
    
    with open(file_path, 'rb') as f:
        files = {"file": f}
        response = requests.post(url, headers=headers, data=data, files=files)
    
    return response.content  # CSV file content

# Get user translations
def get_translations(translation_type="all", limit=50, offset=0):
    url = f"{BASE_URL}/api/translations/"
    params = {
        "translation_type": translation_type,
        "limit": limit,
        "offset": offset
    }
    
    response = requests.get(url, headers=headers, params=params)
    return response.json()

# Example usage
result = translate_text(
    text="Hello, how are you?",
    source_lang="english",
    target_langs=["spanish", "french"],
    style="formal"
)
print(result)
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const BASE_URL = 'https://your-api-domain.com';
const API_KEY = 'your-api-key-here';

const headers = {
    'X-API-Key': API_KEY
};

// Text Translation
async function translateText(text, sourceLang, targetLangs = null, style = null) {
    const url = `${BASE_URL}/translate-text/`;
    
    const formData = new FormData();
    formData.append('text', text);
    formData.append('source_language', sourceLang);
    
    if (targetLangs) {
        targetLangs.forEach(lang => formData.append('target_languages', lang));
    }
    
    if (style) {
        formData.append('translation_style', style);
    }
    
    try {
        const response = await axios.post(url, formData, {
            headers: {
                ...headers,
                ...formData.getHeaders()
            }
        });
        return response.data;
    } catch (error) {
        throw new Error(`Translation failed: ${error.response.data.detail}`);
    }
}

// CSV Translation
async function translateCsv(filePath, sourceLang, targetLangs = null) {
    const url = `${BASE_URL}/translate-csv/`;
    
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));
    formData.append('source_language', sourceLang);
    
    if (targetLangs) {
        targetLangs.forEach(lang => formData.append('target_languages', lang));
    }
    
    try {
        const response = await axios.post(url, formData, {
            headers: {
                ...headers,
                ...formData.getHeaders()
            },
            responseType: 'arraybuffer'
        });
        return response.data;
    } catch (error) {
        throw new Error(`CSV translation failed: ${error.response.data.detail}`);
    }
}

// Example usage
translateText('Hello world', 'english', ['spanish', 'french'])
    .then(result => console.log(result))
    .catch(error => console.error(error));
```

### cURL

```bash
# Text Translation
curl -X POST "https://your-api-domain.com/translate-text/" \
  -H "X-API-Key: your-api-key-here" \
  -F "text=Hello world, how are you?" \
  -F "source_language=english" \
  -F "target_languages=spanish" \
  -F "target_languages=french" \
  -F "translation_style=formal"

# CSV Translation
curl -X POST "https://your-api-domain.com/translate-csv/" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@your-file.csv" \
  -F "source_language=english" \
  -F "target_languages=spanish" \
  --output translated_file.csv

# Get translations list
curl -X GET "https://your-api-domain.com/api/translations/?limit=10&offset=0" \
  -H "X-API-Key: your-api-key-here"

# Get specific translation
curl -X GET "https://your-api-domain.com/api/translation/123?translation_type=text" \
  -H "X-API-Key: your-api-key-here"
```

### PHP

```php
<?php

class TranslationAPI {
    private $baseUrl;
    private $apiKey;
    
    public function __construct($baseUrl, $apiKey) {
        $this->baseUrl = $baseUrl;
        $this->apiKey = $apiKey;
    }
    
    private function getHeaders() {
        return [
            'X-API-Key: ' . $this->apiKey
        ];
    }
    
    public function translateText($text, $sourceLang, $targetLangs = null, $style = null) {
        $url = $this->baseUrl . '/translate-text/';
        
        $postData = [
            'text' => $text,
            'source_language' => $sourceLang
        ];
        
        if ($targetLangs) {
            $postData['target_languages'] = $targetLangs;
        }
        
        if ($style) {
            $postData['translation_style'] = $style;
        }
        
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);
        curl_setopt($ch, CURLOPT_HTTPHEADER, $this->getHeaders());
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        if ($httpCode !== 200) {
            throw new Exception('Translation failed: ' . $response);
        }
        
        return json_decode($response, true);
    }
    
    public function getTranslations($type = 'all', $limit = 50, $offset = 0) {
        $url = $this->baseUrl . '/api/translations/?' . http_build_query([
            'translation_type' => $type,
            'limit' => $limit,
            'offset' => $offset
        ]);
        
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_HTTPHEADER, $this->getHeaders());
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        
        $response = curl_exec($ch);
        curl_close($ch);
        
        return json_decode($response, true);
    }
}

// Example usage
$api = new TranslationAPI('https://your-api-domain.com', 'your-api-key-here');

$result = $api->translateText(
    'Hello world',
    'english',
    ['spanish', 'french'],
    'formal'
);

print_r($result);
?>
```

## Rate Limiting & Costs

### Usage Limits

- **Text Translations**: No hard limit, but usage tracked for billing
- **CSV Translations**: File size limited to 10MB
- **API Calls**: Rate limited to prevent abuse

### Cost Structure

- **Text Translation**: Based on OpenAI API usage + service margin
- **CSV Translation**: Calculated per batch, includes processing costs
- **Character Limits**: Single text limited to 5000 characters
- **CSV Row Limits**: No hard limit, but large files may timeout

### Best Practices

1. **Batch Processing**: Use CSV endpoint for multiple texts
2. **Language Selection**: Specify target languages to reduce costs
3. **Error Handling**: Always implement proper error handling
4. **API Key Security**: Never expose API keys in client-side code
5. **Caching**: Cache translations to avoid duplicate API calls

## Support

For technical support or questions about the API:

- Check the error response for detailed error messages
- Ensure your API key is valid and active
- Verify that language codes are correct
- Check file format for CSV uploads

For billing or account issues, contact your administrator.

---

**Last Updated**: January 2024  
**API Version**: 1.0  
**OpenAI Model**: GPT-4 Turbo