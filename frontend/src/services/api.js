const API_BASE_URL = 'http://localhost:8000';

/**
 * Helper function to handle API errors
 * @param {Response} response - The fetch Response object
 * @returns {Promise<Object>} The parsed JSON response if successful
 * @throws {Error} With error details if the response was not successful
 */
const handleApiResponse = async (response) => {
  if (!response.ok) {
    let errorMessage = `HTTP error! Status: ${response.status}`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch (e) {
      // If we can't parse the error as JSON, use the default message
    }
    throw new Error(errorMessage);
  }
  return await response.json();
};

/**
 * Gets a list of available Ollama models
 * @returns {Promise<Array<string>>} List of available model names
 */
export const getOllamaModels = async () => {
  try {
    // Query the Ollama API directly
    const response = await fetch('http://localhost:11434/api/tags');
    
    if (response.ok) {
      const data = await response.json();
      if (data && data.models) {
        // Extract model names from the response
        return data.models.map(model => model.name);
      }
    }
    
    // Fallback to a backend endpoint if direct access fails
    const backupResponse = await fetch(`${API_BASE_URL}/api/query/models`);
    return await handleApiResponse(backupResponse);
  } catch (error) {
    console.error('Error fetching Ollama models:', error);
    // Return a default model list in case of error
    return ['llama3.2:latest'];
  }
};

/**
 * Fetches the hello message from the backend API
 * @returns {Promise<Object>} The response data as a JSON object
 * @throws {Error} If the API call fails
 */
export const fetchHelloMessage = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/hello`);
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Error fetching hello message:', error);
    throw error;
  }
};

/**
 * Uploads a document to the backend API
 * @param {File} file - The file to upload
 * @returns {Promise<Object>} The response data as a JSON object
 * @throws {Error} If the API call fails
 */
export const uploadDocument = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    console.log(`Uploading file: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);
    
    const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Error uploading document:', error);
    throw error;
  }
};

/**
 * Uploads a document to the backend API asynchronously
 * @param {File} file - The file to upload
 * @returns {Promise<Object>} The response data as a JSON object with doc_id
 * @throws {Error} If the API call fails
 */
export const uploadDocumentAsync = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    console.log(`Starting async upload for: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);
    
    const response = await fetch(`${API_BASE_URL}/api/documents/upload/async`, {
      method: 'POST',
      body: formData,
    });
    
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Error starting async document upload:', error);
    throw error;
  }
};

/**
 * Gets the current processing status for a document
 * @param {string} docId - The document ID
 * @returns {Promise<Object>} The status data as a JSON object
 * @throws {Error} If the API call fails
 */
export const getDocumentStatus = async (docId) => {
  try {
    console.log(`Polling status for document: ${docId}`);
    const response = await fetch(`${API_BASE_URL}/api/documents/status/${docId}`);
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Error fetching document status:', error);
    throw error;
  }
};

/**
 * Creates an EventSource to stream processing status updates
 * @param {string} docId - The document ID
 * @returns {EventSource} An EventSource object
 */
export const streamDocumentStatus = (docId) => {
  try {
    console.log(`Creating event source for document: ${docId}`);
    const eventSource = new EventSource(`${API_BASE_URL}/api/documents/status/stream/${docId}`);
    
    // Add error listener to handle connection issues
    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSource.close();
    };
    
    return eventSource;
  } catch (error) {
    console.error('Error creating EventSource:', error);
    throw new Error('Failed to connect to status stream');
  }
};

/**
 * Submits a query to the backend API
 * @param {string} question - The question to ask
 * @param {Object} modelConfig - Configuration for the LLM model
 * @returns {Promise<Object>} The response data as a JSON object
 * @throws {Error} If the API call fails
 */
export const submitQuery = async (question, modelConfig = {}) => {
  try {
    // Log the query request
    console.log(`Submitting query: "${question}"`);
    console.log(`Using model config:`, modelConfig);
    
    // Start time for performance tracking
    const startTime = performance.now();
    
    const response = await fetch(`${API_BASE_URL}/api/query/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        question,
        config_for_model: modelConfig  // Updated to match the backend model field name
      }),
    });
    
    // Calculate round-trip time
    const requestTime = performance.now() - startTime;
    console.log(`Query round-trip time: ${requestTime.toFixed(2)}ms`);
    
    const result = await handleApiResponse(response);
    
    // Log detailed response information
    console.log(`Query response received:
- Answer length: ${result.answer ? result.answer.length : 0} characters
- Response time: ${result.query_time_ms}ms
- Model used: ${result.model}
- Sources: ${result.num_sources}
`);
    
    return result;
  } catch (error) {
    console.error(`Error submitting query "${question}":`, error);
    // Return a structured error to be displayed in the UI
    return {
      error: true,
      message: error.message || 'Failed to get a response from the server',
      answer: 'Error: ' + (error.message || 'An unexpected error occurred'),
      query_time_ms: 0,
      sources: [],
      num_sources: 0
    };
  }
};

/**
 * Gets the current status of the LLM service
 * @returns {Promise<Object>} The status data as a JSON object
 * @throws {Error} If the API call fails
 */
export const getLLMStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/query/status`);
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Error fetching LLM status:', error);
    throw error;
  }
};

/**
 * Creates an EventSource for streaming query responses
 * @param {string} question - The question to ask
 * @param {Object} modelConfig - Configuration for the LLM model
 * @returns {Object} An object containing the EventSource and methods to manage it
 */


/**
 * Submits a query to stream from the backend using fetch with streaming
 * @param {string} question - The question to ask
 * @returns {Promise<ReadableStream>} A readable stream of the response
 * @throws {Error} If the API call fails
 */
export const fetchStreamingQuery = async (question) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/query/stream/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
    }
    
    return response.body;
  } catch (error) {
    console.error('Error setting up streaming query:', error);
    throw error;
  }
};

/**
 * Gets a list of all documents in the system
 * @returns {Promise<Array>} List of document objects
 * @throws {Error} If the API call fails
 */
export const getDocuments = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/documents`);
    if (!response.ok) {
      throw new Error('Failed to fetch documents');
    }
    const data = await response.json();
    return data.documents || [];  // Access the documents array from the response
  } catch (error) {
    console.error('Error fetching documents:', error);
    throw error;
  }
};

/**
 * Gets details for a specific document
 * @param {number} documentId - The document ID
 * @returns {Promise<Object>} Document details
 * @throws {Error} If the API call fails
 */
export const getDocumentById = async (documentId) => {
  try {
    console.log(`Fetching document details for ID: ${documentId}`);
    const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}`);
    return await handleApiResponse(response);
  } catch (error) {
    console.error(`Error fetching document ${documentId}:`, error);
    throw error;
  }
};

/**
 * Deletes a document from the system
 * @param {number} documentId - The document ID to delete
 * @returns {Promise<Object>} Response with success message
 * @throws {Error} If the API call fails
 */
export const deleteDocument = async (documentId) => {
  try {
    console.log(`Deleting document with ID: ${documentId}`);
    const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}`, {
      method: 'DELETE',
    });
    return await handleApiResponse(response);
  } catch (error) {
    console.error(`Error deleting document ${documentId}:`, error);
    throw error;
  }
};