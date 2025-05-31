const API_BASE_URL = 'http://localhost:8000';

// Connection management
let isServerAvailable = true;
let lastConnectionCheck = 0;
const CONNECTION_CHECK_INTERVAL = 30000; // 30 seconds

/**
 * Check if the backend server is available
 * @returns {Promise<boolean>} True if server is available
 */
const checkServerConnection = async () => {
  const now = Date.now();
  
  // Only check connection every 30 seconds to avoid spam
  if (now - lastConnectionCheck < CONNECTION_CHECK_INTERVAL && isServerAvailable) {
    return isServerAvailable;
  }
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
    
    const response = await fetch(`${API_BASE_URL}/api/hello`, {
      method: 'GET',
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    isServerAvailable = response.ok;
    lastConnectionCheck = now;
    
    if (!isServerAvailable) {
      console.warn('Backend server returned non-OK status:', response.status);
    }
    
    return isServerAvailable;
  } catch (error) {
    isServerAvailable = false;
    lastConnectionCheck = now;
    console.error('Backend server connection check failed:', error.message);
    return false;
  }
};

/**
 * Helper function to handle API errors
 * @param {Response} response - The fetch Response object
 * @returns {Promise<Object>} The parsed JSON response if successful
 * @throws {Error} With error details if the response was not successful
 */
const handleApiResponse = async (response) => {
  if (!response.ok) {
    let errorMessage = `HTTP error! Status: ${response.status}`;
    let errorDetails = {};
    
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
        errorDetails = errorData;
      }
    } catch (e) {
      // If we can't parse the error as JSON, provide more context
      errorDetails.statusText = response.statusText;
      errorDetails.url = response.url;
    }
    
    // Add specific error handling for common issues
    if (response.status === 0 || response.status >= 500) {
      errorMessage = 'Failed to connect to the backend server. Please check if the server is running.';
    } else if (response.status === 404) {
      errorMessage = 'API endpoint not found. Please check the backend configuration.';
    } else if (response.status === 429) {
      errorMessage = 'Too many requests. Please wait a moment and try again.';
    }
    
    const error = new Error(errorMessage);
    error.status = response.status;
    error.details = errorDetails;
    throw error;
  }
  return await response.json();
};

/**
 * Enhanced fetch wrapper with automatic retry and better error handling
 * @param {string} url - The URL to fetch
 * @param {Object} options - Fetch options
 * @param {number} retries - Number of retries (default: 1)
 * @returns {Promise<Response>} The fetch response
 */
const fetchWithRetry = async (url, options = {}, retries = 1) => {
  // Check server connection first
  const serverAvailable = await checkServerConnection();
  if (!serverAvailable) {
    throw new Error('Backend server is not available. Please ensure the server is running on http://localhost:8000');
  }
  
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {      // Add timeout to prevent hanging requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 90000); // Increased to 90 seconds for slower models
      
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      return response;
      
    } catch (error) {
      console.warn(`Attempt ${attempt + 1} failed for ${url}:`, error.message);
      
      // Mark server as unavailable for connection errors
      if (error.name === 'AbortError' || error.message.includes('fetch')) {
        isServerAvailable = false;
      }
      
      // If this is the last attempt, throw the error
      if (attempt === retries) {
        if (error.name === 'AbortError') {
          throw new Error('Request timed out. The server may be overloaded.');
        }
        throw new Error(`Failed to connect to the backend server: ${error.message}`);
      }
      
      // Wait before retrying (exponential backoff)
      await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
    }
  }
};

/**
 * Get available LLM models
 * @returns {Promise<Array>} List of available models grouped by provider
 */
export const getAvailableLLMModels = async () => {
  try {
    const serverAvailable = await checkServerConnection();
    if (!serverAvailable) {
      throw new Error('Backend server is not available. Please ensure it is running on http://localhost:8000');
    }

    const response = await fetch(`${API_BASE_URL}/api/query/models`);
    const data = await handleApiResponse(response);
    
    if (data && data.models) {
      // Group models by provider for easier UI handling
      const groupedModels = {
        ollama: [],
        openai: [],
        gemini: [],
        huggingface: []
      };
      
      data.models.forEach(model => {
        if (groupedModels[model.provider]) {
          groupedModels[model.provider].push(model);
        }
      });
      
      return groupedModels;
    }
    
    return {
      ollama: [],
      openai: [],
      gemini: [],
      huggingface: []
    };
  } catch (error) {
    console.error('Error fetching available LLM models:', error);    throw error;
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
 * Uploads a document to the backend API with enhanced error handling
 * @param {File} file - The file to upload
 * @returns {Promise<Object>} The response data as a JSON object
 * @throws {Error} If the API call fails
 */
export const uploadDocument = async (file) => {
  try {
    // Check server connection first
    const serverAvailable = await checkServerConnection();
    if (!serverAvailable) {
      throw new Error('Backend server is not available. Please ensure the server is running on http://localhost:8000');
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    console.log(`Uploading file: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);
    
    const response = await fetchWithRetry(`${API_BASE_URL}/api/documents/upload`, {
      method: 'POST',
      body: formData,
    }, 1); // Allow 1 retry for uploads
    
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Error uploading document:', error.message);
    
    let userMessage = 'Failed to upload document';
    if (error.message.includes('Backend server is not available')) {
      userMessage = 'Cannot connect to server. Please check if the backend is running.';
    } else if (error.message.includes('timed out')) {
      userMessage = 'Upload timed out. Please try again with a smaller file.';
    } else if (error.status === 413) {
      userMessage = 'File is too large. Please try a smaller file.';
    } else if (error.status === 415) {
      userMessage = 'File type not supported. Please upload a PDF or CSV file.';
    }
    
    const enhancedError = new Error(userMessage);
    enhancedError.originalError = error;
    throw enhancedError;
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
    
    const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
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
 * Submits a query to the backend API with enhanced error handling, retry logic, and BGE reranking
 * @param {string} question - The question to ask
 * @param {Object} modelConfig - Configuration for the LLM model
 * @param {string} searchStrategy - Search strategy: 'semantic', 'hybrid', or 'keyword'
 * @param {number} maxSources - Maximum number of source documents to retrieve
 * @param {boolean} useReranking - Whether to use BGE reranking (default: true for +23.86% MAP improvement)
 * @param {string} rerankerModel - BGE reranker model to use
 * @returns {Promise<Object>} The response data as a JSON object with enhanced relevance scoring
 * @throws {Error} If the API call fails after retries
 */
export const submitQuery = async (question, modelConfig = {}, searchStrategy = 'hybrid', maxSources = 5, useReranking = true, rerankerModel = 'BAAI/bge-reranker-base') => {
  try {
    // Log the query request
    console.log(`Submitting query: "${question}"`);
    console.log(`Using model config:`, modelConfig);
    console.log(`Search strategy: ${searchStrategy}, Max sources: ${maxSources}`);
    console.log(`BGE Reranking enabled: ${useReranking}, Model: ${rerankerModel}`);
    
    // Check server connection first
    const serverAvailable = await checkServerConnection();
    if (!serverAvailable) {
      throw new Error('Backend server is not available. Please ensure the server is running on http://localhost:8000');
    }
    
    // Start time for performance tracking
    const startTime = performance.now();
      const response = await fetchWithRetry(`${API_BASE_URL}/api/query/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        question,
        config_for_model: modelConfig,
        search_strategy: searchStrategy,
        max_sources: maxSources,
        use_reranking: useReranking,
        reranker_model: rerankerModel
      }),
    }, 2); // Allow 2 retries for queries
    
    // Calculate round-trip time
    const requestTime = performance.now() - startTime;
    console.log(`Query round-trip time: ${requestTime.toFixed(2)}ms`);
    
    const result = await handleApiResponse(response);
    
    // Log detailed response information
    console.log(`Query response received:
- Answer length: ${result.answer ? result.answer.length : 0} characters
- Response time: ${result.query_time_ms}ms
- Retrieval time: ${result.retrieval_time_ms}ms
- LLM time: ${result.llm_time_ms}ms
- Model used: ${result.model}
- Sources: ${result.num_sources}
- Search strategy: ${result.search_strategy}
- Average relevance: ${result.average_relevance}
- Top relevance: ${result.top_relevance}
`);
    
    return result;
    
  } catch (error) {
    console.error(`Error submitting query "${question}":`, error.message);
    
    // Provide detailed error information based on error type
    let userMessage = 'Failed to get a response from the server';
    
    if (error.message.includes('Backend server is not available')) {
      userMessage = 'Cannot connect to the backend server. Please check if the server is running on http://localhost:8000';
    } else if (error.message.includes('timed out')) {
      userMessage = 'The request timed out. The server may be overloaded. Please try again.';
    } else if (error.status === 500) {
      userMessage = 'Server error occurred while processing your query. Please try again or contact support.';
    } else if (error.status === 400) {
      userMessage = 'Invalid query format. Please check your question and try again.';
    } else if (error.message.includes('fetch')) {
      userMessage = 'Network connection error. Please check your internet connection and server status.';
    }
    
    // Return a structured error to be displayed in the UI
    return {
      error: true,
      message: userMessage,
      answer: `Error: ${userMessage}`,
      query_time_ms: 0,
      sources: [],
      num_sources: 0,
      search_strategy: searchStrategy,
      technical_details: error.message // For debugging
    };
  }
};

/**
 * Gets the current status of the LLM service with enhanced error handling
 * @returns {Promise<Object>} The status data as a JSON object
 * @throws {Error} If the API call fails
 */
export const getLLMStatus = async () => {
  try {
    // Use shorter timeout for status checks
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout for status
    
    const response = await fetch(`${API_BASE_URL}/api/query/status`, {
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      // Don't throw for status checks, return a default status instead
      console.warn(`LLM status check failed with status ${response.status}`);
      return {
        is_processing: false,
        total_queries: 0,
        successful_queries: 0,
        cache_size: 0,
        last_model_used: null,
        last_query_time: null,
        status: 'unknown'
      };
    }
    
    return await response.json();
  } catch (error) {
    console.warn('Error fetching LLM status (non-critical):', error.message);
    // Return a default status object instead of throwing
    return {
      is_processing: false,
      total_queries: 0,
      successful_queries: 0,
      cache_size: 0,
      last_model_used: null,
      last_query_time: null,
      status: 'connection_error'
    };
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

/**
 * Gets list of images extracted from a document
 * @param {string} documentId - The document ID
 * @returns {Promise<Object>} Response with document images metadata
 * @throws {Error} If the API call fails
 */
export const getDocumentImages = async (documentId) => {
  try {
    console.log(`Fetching images for document ID: ${documentId}`);
    const response = await fetchWithRetry(`${API_BASE_URL}/api/documents/${documentId}/images`);
    return await handleApiResponse(response);
  } catch (error) {
    console.error(`Error fetching images for document ${documentId}:`, error);
    throw error;
  }
};

/**
 * Gets the URL for a specific document image
 * @param {string} documentId - The document ID
 * @param {string} imageFilename - The image filename
 * @returns {string} The complete URL to the image
 */
export const getDocumentImageUrl = (documentId, imageFilename) => {
  return `${API_BASE_URL}/api/documents/${documentId}/images/${imageFilename}`;
};

/**
 * Downloads a specific image from a document
 * @param {string} documentId - The document ID
 * @param {string} imageFilename - The image filename
 * @returns {Promise<Blob>} The image blob data
 * @throws {Error} If the API call fails
 */
export const downloadDocumentImage = async (documentId, imageFilename) => {
  try {
    console.log(`Downloading image ${imageFilename} for document ID: ${documentId}`);
    const response = await fetchWithRetry(`${API_BASE_URL}/api/documents/${documentId}/images/${imageFilename}`);
    
    if (!response.ok) {
      throw new Error(`Failed to download image: ${response.status} ${response.statusText}`);
    }
    
    return await response.blob();
  } catch (error) {
    console.error(`Error downloading image ${imageFilename} for document ${documentId}:`, error);
    throw error;
  }
};

/**
 * Submits a query with intelligent decomposition to the backend API
 * @param {string} question - The question to ask
 * @param {Object} modelConfig - Configuration for the LLM model
 * @param {string} searchStrategy - Search strategy: 'semantic', 'hybrid', or 'keyword'
 * @param {number} maxSources - Maximum number of source documents to retrieve
 * @param {boolean} useDecomposition - Whether to enable query decomposition
 * @returns {Promise<Object>} The decomposed response data as a JSON object
 * @throws {Error} If the API call fails after retries
 */
export const submitDecomposedQuery = async (question, modelConfig = {}, searchStrategy = 'hybrid', maxSources = 5, useDecomposition = true, useReranking = true, rerankerModel = 'BAAI/bge-reranker-base') => {
  try {
    // Log the decomposed query request
    console.log(`Submitting decomposed query: "${question}"`);
    console.log(`Using model config:`, modelConfig);
    console.log(`Search strategy: ${searchStrategy}, Max sources: ${maxSources}`);    console.log(`Decomposition enabled: ${useDecomposition}`);
    console.log(`BGE Reranking enabled: ${useReranking}, Model: ${rerankerModel}`);
    
    // Check server connection first
    const serverAvailable = await checkServerConnection();
    if (!serverAvailable) {
      throw new Error('Backend server is not available. Please ensure the server is running on http://localhost:8000');
    }
    
    // Start time for performance tracking
    const startTime = performance.now();
    
    const response = await fetchWithRetry(`${API_BASE_URL}/api/query/decomposed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },      body: JSON.stringify({ 
        question,
        config_for_model: modelConfig,
        search_strategy: searchStrategy,
        max_sources: maxSources,
        use_decomposition: useDecomposition,
        use_reranking: useReranking,
        reranker_model: rerankerModel
      }),
    }, 2); // Allow 2 retries for queries
    
    // Calculate round-trip time
    const requestTime = performance.now() - startTime;
    console.log(`Decomposed query round-trip time: ${requestTime.toFixed(2)}ms`);
    
    const result = await handleApiResponse(response);
    
    // Log detailed response information
    console.log(`Decomposed query response received:
- Original query: "${result.original_query}"
- Decomposed: ${result.is_decomposed}
- Sub-queries: ${result.sub_queries ? result.sub_queries.length : 0}
- Final answer length: ${result.final_answer ? result.final_answer.length : 0} characters
- Total response time: ${result.total_query_time_ms}ms
- Decomposition time: ${result.decomposition_time_ms}ms
- Synthesis time: ${result.synthesis_time_ms}ms
- Model used: ${result.model}
- Total sources: ${result.total_sources}
- Search strategy: ${result.search_strategy}
- Average relevance: ${result.average_relevance}
`);
    
    // Log sub-query details
    if (result.sub_results && result.sub_results.length > 0) {
      console.log('Sub-query breakdown:');
      result.sub_results.forEach((subResult, index) => {
        console.log(`  ${index + 1}. "${subResult.sub_query}" -> ${subResult.answer.length} chars, ${subResult.num_sources} sources, ${subResult.processing_time_ms}ms`);
      });
    }
    
    return result;
    
  } catch (error) {
    console.error(`Error submitting decomposed query "${question}":`, error.message);
    
    // Provide detailed error information based on error type
    let userMessage = 'Failed to get a response from the server';
    
    if (error.message.includes('Backend server is not available')) {
      userMessage = 'Cannot connect to the backend server. Please check if the server is running on http://localhost:8000';
    } else if (error.message.includes('timed out')) {
      userMessage = 'The request timed out. The server may be overloaded. Please try again.';
    } else if (error.status === 500) {
      userMessage = 'Server error occurred while processing your query. Please try again or contact support.';
    } else if (error.status === 400) {
      userMessage = 'Invalid query format. Please check your question and try again.';
    } else if (error.message.includes('fetch')) {
      userMessage = 'Network connection error. Please check your internet connection and server status.';
    }
    
    // Return a structured error to be displayed in the UI
    return {
      error: true,
      message: userMessage,
      original_query: question,
      is_decomposed: false,
      sub_queries: [question],
      sub_results: [],
      final_answer: `Error: ${userMessage}`,
      total_query_time_ms: 0,
      total_sources: 0,
      search_strategy: searchStrategy,
      technical_details: error.message // For debugging
    };
  }
};

/**
 * Gets all chunks (actual content) for a specific document
 * @param {string} docId - The document ID
 * @returns {Promise<Object>} Response with document chunks and content
 * @throws {Error} If the API call fails
 */
export const getDocumentChunks = async (docId) => {
  try {
    console.log(`Fetching chunks for document ID: ${docId}`);
    const response = await fetchWithRetry(`${API_BASE_URL}/api/documents/${docId}/chunks`);
    return await handleApiResponse(response);
  } catch (error) {
    console.error(`Error fetching chunks for document ${docId}:`, error);
    throw error;
  }
};

/**
 * Gets specific chunk content for a document
 * @param {string} docId - The document ID
 * @param {number} chunkIndex - The chunk index
 * @returns {Promise<Object>} The chunk data with content
 * @throws {Error} If the API call fails
 */
export const getDocumentChunk = async (docId, chunkIndex) => {
  try {
    console.log(`Fetching chunk ${chunkIndex} for document ID: ${docId}`);
    const response = await fetchWithRetry(`${API_BASE_URL}/api/documents/${docId}/chunks/${chunkIndex}`);
    return await handleApiResponse(response);
  } catch (error) {
    console.error(`Error fetching chunk ${chunkIndex} for document ${docId}:`, error);
    throw error;
  }
};

/**
 * Get complete system configuration
 * @returns {Promise<Object>} Complete configuration object
 */
export const getSystemConfiguration = async () => {
  if (!await checkServerConnection()) {
    throw new Error('Backend server is not available');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/config/`);
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to get system configuration:', error);
    throw error;
  }
};

/**
 * Update embedding model
 * @param {string} modelName - Name of the embedding model to use
 * @returns {Promise<Object>} Update result
 */
export const updateEmbeddingModel = async (modelName) => {
  if (!await checkServerConnection()) {
    throw new Error('Backend server is not available');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/config/embedding/model`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model_name: modelName }),
    });
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to update embedding model:', error);
    throw error;
  }
};

/**
 * Update LLM provider
 * @param {string} provider - LLM provider name
 * @returns {Promise<Object>} Update result
 */
export const updateLLMProvider = async (provider) => {
  if (!await checkServerConnection()) {
    throw new Error('Backend server is not available');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/config/llm/provider`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ provider }),
    });
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to update LLM provider:', error);
    throw error;
  }
};

/**
 * Update LLM model
 * @param {string} model - LLM model name
 * @returns {Promise<Object>} Update result
 */
export const updateLLMModel = async (model) => {
  if (!await checkServerConnection()) {
    throw new Error('Backend server is not available');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/config/llm/model`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model }),
    });
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to update LLM model:', error);
    throw error;
  }
};

/**
 * Update search strategy
 * @param {string} strategy - Search strategy (hybrid, semantic, keyword)
 * @returns {Promise<Object>} Update result
 */
export const updateSearchStrategy = async (strategy) => {
  if (!await checkServerConnection()) {
    throw new Error('Backend server is not available');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/config/search/strategy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ strategy }),
    });
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to update search strategy:', error);
    throw error;
  }
};

/**
 * Update max sources
 * @param {number} maxSources - Maximum number of sources
 * @returns {Promise<Object>} Update result
 */
export const updateMaxSources = async (maxSources) => {
  if (!await checkServerConnection()) {
    throw new Error('Backend server is not available');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/config/search/max-sources`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ max_sources: maxSources }),
    });
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to update max sources:', error);
    throw error;
  }
};

/**
 * Toggle query decomposition
 * @param {boolean} enabled - Whether to enable query decomposition
 * @returns {Promise<Object>} Update result
 */
export const toggleQueryDecomposition = async (enabled) => {
  if (!await checkServerConnection()) {
    throw new Error('Backend server is not available');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/config/search/query-decomposition`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ enabled }),
    });
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to toggle query decomposition:', error);
    throw error;
  }
};

/**
 * Store API key for a provider with persistent backend storage
 * @param {string} provider - Provider name (openai, gemini, huggingface)
 * @param {string} apiKey - The API key to store
 * @returns {Promise<Object>} Storage result
 */
export const storeApiKey = async (provider, apiKey) => {
  try {
    const response = await fetchWithRetry(`${API_BASE_URL}/api/v1/config/api-keys/${provider}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ api_key: apiKey }),
    });
    
    const result = await handleApiResponse(response);
    
    // Also store in session storage as fallback for immediate use
    if (apiKey && apiKey.trim()) {
      sessionStorage.setItem(`api_key_${provider}`, apiKey.trim());
    } else {
      sessionStorage.removeItem(`api_key_${provider}`);
    }
    
    return result;
  } catch (error) {
    console.error('Failed to store API key:', error);
    
    // Fallback to session storage if backend fails
    try {
      if (apiKey && apiKey.trim()) {
        sessionStorage.setItem(`api_key_${provider}`, apiKey.trim());
      } else {
        sessionStorage.removeItem(`api_key_${provider}`);
      }
      
      return { 
        success: true, 
        message: `API key ${apiKey ? 'stored' : 'cleared'} for ${provider} (session only)`,
        fallback: true
      };
    } catch (fallbackError) {
      throw new Error(`Failed to store API key for ${provider}: ${error.message}`);
    }
  }
};

/**
 * Get stored API keys status (without revealing the actual keys)
 * @returns {Promise<Object>} API keys status
 */
export const getApiKeysStatus = async () => {
  try {
    const response = await fetchWithRetry(`${API_BASE_URL}/api/v1/config/api-keys/status`);
    const result = await handleApiResponse(response);
    return result.api_keys;
  } catch (error) {
    console.error('Failed to get API keys status from backend:', error);
    
    // Fallback to session storage if backend fails
    try {
      const status = {
        openai: !!sessionStorage.getItem('api_key_openai'),
        gemini: !!sessionStorage.getItem('api_key_gemini'),
        huggingface: !!sessionStorage.getItem('api_key_huggingface')
      };
      
      return status;
    } catch (fallbackError) {
      console.error('Failed to get API keys status from session storage:', fallbackError);
      // Return default status on error
      return { openai: false, gemini: false, huggingface: false };
    }
  }
};

/**
 * Clear API key for a provider
 * @param {string} provider - Provider name (openai, gemini, huggingface)
 * @returns {Promise<Object>} Clear result
 */
export const clearApiKey = async (provider) => {
  try {
    const response = await fetchWithRetry(`${API_BASE_URL}/api/v1/config/api-keys/${provider}`, {
      method: 'DELETE',
    });
    
    const result = await handleApiResponse(response);
    
    // Also clear from session storage
    sessionStorage.removeItem(`api_key_${provider}`);
    
    return result;
  } catch (error) {
    console.error('Failed to clear API key from backend:', error);
    
    // Fallback to clearing session storage if backend fails
    try {
      sessionStorage.removeItem(`api_key_${provider}`);
      
      return { 
        success: true, 
        message: `API key cleared for ${provider} (session only)`,
        fallback: true
      };
    } catch (fallbackError) {
      throw new Error(`Failed to clear API key for ${provider}: ${error.message}`);
    }
  }
};

/**
 * Get stored API key for a provider (for sending with requests)
 * @param {string} provider - Provider name (openai, gemini, huggingface)
 * @returns {string|null} The API key or null if not found
 */
export const getApiKey = (provider) => {
  try {
    return sessionStorage.getItem(`api_key_${provider}`);
  } catch (error) {
    console.error('Failed to get API key:', error);
    return null;
  }
};

/**
 * Get BGE reranker configuration and available models
 * @returns {Promise<Object>} Reranker configuration with benchmark results
 */
export const getRerankerConfig = async () => {
  try {
    const response = await fetchWithRetry(`${API_BASE_URL}/api/query/reranker/config`);
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to get reranker configuration:', error);
    throw new Error(`Failed to get reranker configuration: ${error.message}`);
  }
};

/**
 * Toggle BGE reranking on/off for the system with persistent storage
 * @param {boolean} enable - Whether to enable reranking
 * @returns {Promise<Object>} Toggle result with status
 */
export const toggleReranking = async (enable) => {
  try {
    const response = await fetchWithRetry(`${API_BASE_URL}/api/v1/config/reranking/toggle`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ enabled: enable }),
    });
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to toggle reranking:', error);
    throw new Error(`Failed to toggle reranking: ${error.message}`);
  }
};

/**
 * Test BGE reranker performance with a query
 * @param {string} question - The test question
 * @param {string} rerankerModel - BGE reranker model to test
 * @returns {Promise<Object>} Comparison of original vs reranked results
 */
export const testReranker = async (question, rerankerModel = 'BAAI/bge-reranker-base') => {
  try {
    const response = await fetchWithRetry(`${API_BASE_URL}/api/query/reranker/test`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        reranker_model: rerankerModel
      }),
    });
    return await handleApiResponse(response);
  } catch (error) {
    console.error('Failed to test reranker:', error);
    throw new Error(`Failed to test reranker: ${error.message}`);
  }
};