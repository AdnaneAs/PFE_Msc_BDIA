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