const API_BASE_URL = 'http://localhost:8000';

/**
 * Fetches the hello message from the backend API
 * @returns {Promise<Object>} The response data as a JSON object
 * @throws {Error} If the API call fails
 */
export const fetchHelloMessage = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/hello`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
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
    
    const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
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
    
    const response = await fetch(`${API_BASE_URL}/api/documents/upload/async`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
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
    const response = await fetch(`${API_BASE_URL}/api/documents/status/${docId}`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
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
 * @returns {Promise<Object>} The response data as a JSON object
 * @throws {Error} If the API call fails
 */
export const submitQuery = async (question) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/query`, {
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
    
    return await response.json();
  } catch (error) {
    console.error('Error submitting query:', error);
    throw error;
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
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching LLM status:', error);
    throw error;
  }
};

/**
 * Creates an EventSource for streaming query responses
 * @param {string} question - The question to ask
 * @returns {EventSource} An EventSource object for streaming the response
 */
export const streamQuery = (question) => {
  try {
    // URL encode the question and add to the URL
    const encodedQuestion = encodeURIComponent(question);
    const eventSource = new EventSource(`${API_BASE_URL}/api/query/stream?question=${encodedQuestion}`);
    
    // Add error handler for EventSource
    eventSource.onerror = (error) => {
      console.error('Query stream error:', error);
      eventSource.close();
    };
    
    return eventSource;
  } catch (error) {
    console.error('Error creating query EventSource:', error);
    throw new Error('Failed to connect to query stream');
  }
};

/**
 * Submits a query to stream from the backend using fetch with streaming
 * @param {string} question - The question to ask
 * @returns {Promise<ReadableStream>} A readable stream of the response
 * @throws {Error} If the API call fails
 */
export const fetchStreamingQuery = async (question) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/query/stream`, {
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