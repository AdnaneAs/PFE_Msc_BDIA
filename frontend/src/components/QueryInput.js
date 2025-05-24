import React, { useState, useEffect, useRef } from 'react';
import { submitQuery, getLLMStatus, streamQuery, getOllamaModels } from '../services/api';

const QueryInput = ({ onQueryResult }) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queryStatus, setQueryStatus] = useState(null);
  const [llmStatus, setLlmStatus] = useState(null);
  const [showLlmStatus, setShowLlmStatus] = useState(false);
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [streamedResponse, setStreamedResponse] = useState('');
  const [streamSources, setStreamSources] = useState([]);
  const eventSourceRef = useRef(null);

  // Model selection states
  const [modelProvider, setModelProvider] = useState('ollama_local');
  const [apiKey, setApiKey] = useState('');
  const [ollamaModels, setOllamaModels] = useState([]);
  const [selectedOllamaModel, setSelectedOllamaModel] = useState('llama3.2:latest');
  
  // Fetch Ollama models on initial load
  useEffect(() => {
    const fetchOllamaModels = async () => {
      try {
        const models = await getOllamaModels();
        if (models && models.length > 0) {
          setOllamaModels(models);
          // Default to first model if available
          if (models.length > 0 && !models.includes(selectedOllamaModel)) {
            setSelectedOllamaModel(models[0]);
          }
        } else {
          setOllamaModels(['llama3.2:latest']);
        }
      } catch (err) {
        console.error("Failed to fetch Ollama models:", err);
        setOllamaModels(['llama3.2:latest']);
      }
    };
    
    fetchOllamaModels();
  }, []);

  // Fetch LLM status initially and after queries
  useEffect(() => {
    fetchLlmStatus();
    
    // Set up interval to fetch status every 10 seconds
    const intervalId = setInterval(fetchLlmStatus, 10000);
    
    // Clean up interval
    return () => clearInterval(intervalId);
  }, []);
  
  const fetchLlmStatus = async () => {
    try {
      const status = await getLLMStatus();
      setLlmStatus(status);
    } catch (err) {
      console.error("Failed to fetch LLM status:", err);
      // Don't set error state here to avoid disrupting the UI
    }
  };

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  const handleQuestionChange = (e) => {
    setQuestion(e.target.value);
    setError(null);
    setQueryStatus(null);
  };

  const getCurrentModelConfig = () => {
    // Create model config based on selected provider
    if (modelProvider === 'ollama_local') {
      return {
        provider: 'ollama',
        model: selectedOllamaModel
      };
    } else if (modelProvider === 'openai') {
      return {
        provider: 'openai',
        model: 'gpt-3.5-turbo',
        api_key: apiKey
      };
    } else if (modelProvider === 'gemini') {
      return {
        provider: 'gemini',
        model: 'gemini-pro',
        api_key: apiKey
      };
    }
    
    // Default fallback
    return {
      provider: 'ollama',
      model: 'llama3.2:latest'
    };
  };

  const handleSubmitStreaming = async (e) => {
    e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }
    
    // Check if API key is provided when needed
    if ((modelProvider === 'openai' || modelProvider === 'gemini') && !apiKey) {
      setError(`Please enter an API key for ${modelProvider === 'openai' ? 'OpenAI' : 'Gemini'}`);
      return;
    }
    
    setLoading(true);
    setError(null);
    setStreamedResponse('');
    setStreamSources([]);
    
    // Update status to show steps
    setQueryStatus({
      state: 'searching',
      message: 'Searching for relevant documents...'
    });
    
    try {
      // Close any existing stream
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      
      // Get current model configuration
      const modelConfig = getCurrentModelConfig();
      
      // Create the event source for streaming
      const eventSource = streamQuery(question, modelConfig);
      eventSourceRef.current = eventSource;
      
      // Handle incoming chunks
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Handle metadata message (sent at the beginning)
          if (data.metadata) {
            setStreamSources(data.sources || []);
            setQueryStatus({
              state: 'generating',
              message: `Found ${data.num_sources} relevant documents. Generating answer...`,
              retrievalTime: data.retrieval_time_ms
            });
            
            // Create and pass a partial result to display sources immediately
            const partialResult = {
              answer: '',
              sources: data.sources,
              num_sources: data.num_sources,
              retrieval_time_ms: data.retrieval_time_ms,
              streaming: true
            };
            onQueryResult(partialResult);
          }          
          // Handle token updates
          else if (data.token) {
            // Update the streamed response in state
            const newResponse = streamedResponse + data.token;
            setStreamedResponse(newResponse);
            
            // Log the token and current accumulated response for debugging
            console.log("Received token:", data.token);
            console.log("Current accumulated response length:", newResponse.length);
            
            // Update the result with the current accumulated response
            const updatedResult = {
              answer: newResponse,
              sources: streamSources,
              num_sources: streamSources.length,
              streaming: true,
              retrieval_time_ms: queryStatus?.retrievalTime,
            };
            onQueryResult(updatedResult);
          }
          // Handle completion message
          else if (data.complete) {
            setQueryStatus({
              state: 'success',
              message: 'Answer generated successfully!',
              queryTime: data.query_time_ms
            });
            
            // Create the final result object with the latest accumulated response
            // This ensures we're using the most up-to-date response
            const finalResponse = streamedResponse + (data.final_token || '');
            setStreamedResponse(finalResponse);
            
            console.log("Streaming complete. Final answer length:", finalResponse.length);
            
            const finalResult = {
              answer: finalResponse,
              sources: streamSources,
              num_sources: streamSources.length,
              query_time_ms: data.query_time_ms,
              retrieval_time_ms: queryStatus?.retrievalTime,
              streaming: false
            };
            
            // Make sure the final result is sent to the parent component
            onQueryResult(finalResult);
            
            // Close the event source
            eventSource.close();
            eventSourceRef.current = null;
            setLoading(false);
            
            // Fetch updated LLM status after query completion
            fetchLlmStatus();
          }
          // Handle error message
          else if (data.error) {
            setError(data.message || 'An error occurred while streaming the response');
            setQueryStatus({
              state: 'error',
              message: 'Failed to get a streaming answer'
            });
            
            // Close the event source
            eventSource.close();
            eventSourceRef.current = null;
            setLoading(false);
            
            // Fallback to normal query
            if (streamedResponse === '') {
              handleSubmitNormal(null, true);
            }
          }
        } catch (parseError) {
          console.error('Error parsing stream data:', parseError, event.data);
          setError('Error parsing stream data');
          eventSource.close();
          eventSourceRef.current = null;
          setLoading(false);
        }
      };
      
      // Handle errors in the stream
      eventSource.onerror = (err) => {
        console.error('Stream error:', err);
        setError('Error connecting to streaming service - falling back to regular query');
        
        // Close the event source
        eventSource.close();
        eventSourceRef.current = null;
        
        // Fall back to normal query if we haven't received any response yet
        if (streamedResponse === '') {
          handleSubmitNormal(null, true);
        } else {
          setLoading(false);
        }
      };
    } catch (err) {
      console.error('Failed to set up streaming:', err);
      setError('Failed to set up streaming - falling back to regular query');
      
      // Fall back to normal query
      handleSubmitNormal(null, true);
    }
  };

  const handleSubmitNormal = async (e, isRetry = false) => {
    if (e) e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }
    
    // Check if API key is provided when needed
    if ((modelProvider === 'openai' || modelProvider === 'gemini') && !apiKey) {
      setError(`Please enter an API key for ${modelProvider === 'openai' ? 'OpenAI' : 'Gemini'}`);
      return;
    }
    
    if (!isRetry) {
      setLoading(true);
      setError(null);
    }
    
    // Update status to show steps
    setQueryStatus({
      state: 'searching',
      message: 'Searching for relevant documents...'
    });
    
    try {
      // Start timestamp for performance measurement
      const startTime = Date.now();
      
      // Get current model configuration
      const modelConfig = getCurrentModelConfig();
      
      // Send the query to the backend
      const result = await submitQuery(question, modelConfig);
      
      // Update status
      setQueryStatus({
        state: 'success',
        message: 'Answer generated successfully!',
        queryTime: result.query_time_ms
      });
      
      console.log("Normal query complete. Result:", result);
      
      // Explicitly ensure the answer is passed to the parent component
      if (result) {
        onQueryResult({
          ...result,
          streaming: false
        });
      }
      
      // Fetch updated LLM status after query completion
      fetchLlmStatus();
    } catch (err) {
      console.error('Error submitting query:', err);
      setError(err.message || 'Error submitting query');
      setQueryStatus({
        state: 'error',
        message: 'Failed to get an answer'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    if (streamingEnabled) {
      handleSubmitStreaming(e);
    } else {
      handleSubmitNormal(e);
    }
  };

  // Format time values for display
  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Query Documents</h2>
      
      {/* Model selection UI */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Model Selection</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          <div className="flex items-center">
            <input
              id="ollama-local"
              type="radio"
              value="ollama_local"
              checked={modelProvider === 'ollama_local'}
              onChange={() => setModelProvider('ollama_local')}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor="ollama-local" className="ml-2 block text-sm text-gray-700">
              Ollama (Local)
            </label>
          </div>
          <div className="flex items-center">
            <input
              id="openai"
              type="radio"
              value="openai"
              checked={modelProvider === 'openai'}
              onChange={() => setModelProvider('openai')}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor="openai" className="ml-2 block text-sm text-gray-700">
              OpenAI
            </label>
          </div>
          <div className="flex items-center">
            <input
              id="gemini"
              type="radio"
              value="gemini"
              checked={modelProvider === 'gemini'}
              onChange={() => setModelProvider('gemini')}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor="gemini" className="ml-2 block text-sm text-gray-700">
              Google Gemini
            </label>
          </div>
        </div>
        
        {/* Dynamic model options based on selected provider */}
        {modelProvider === 'ollama_local' && (
          <div className="mb-4">
            <label htmlFor="ollama-model" className="block text-sm font-medium text-gray-700 mb-1">
              Ollama Model
            </label>
            <select
              id="ollama-model"
              value={selectedOllamaModel}
              onChange={(e) => setSelectedOllamaModel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              {ollamaModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
        )}
        
        {/* API Key input for OpenAI or Gemini */}
        {(modelProvider === 'openai' || modelProvider === 'gemini') && (
          <div className="mb-4">
            <label htmlFor="api-key" className="block text-sm font-medium text-gray-700 mb-1">
              {modelProvider === 'openai' ? 'OpenAI' : 'Gemini'} API Key
            </label>
            <input
              id="api-key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={`Enter your ${modelProvider === 'openai' ? 'OpenAI' : 'Gemini'} API key`}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        )}
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-2">
            Ask a question about your documents
          </label>
          <input
            id="question"
            type="text"
            value={question}
            onChange={handleQuestionChange}
            placeholder="e.g., What are the key findings in the audit report?"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <button
              type="submit"
              disabled={loading}
              className={`px-4 py-2 rounded-md font-medium text-white ${
                loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {loading ? 'Getting Answer...' : 'Submit Question'}
            </button>
            
            <div className="flex items-center">
              <label htmlFor="streaming-toggle" className="mr-2 text-sm text-gray-600">
                Streaming
              </label>
              <div className="relative inline-block w-10 mr-2 align-middle select-none">
                <input
                  type="checkbox"
                  name="streaming-toggle"
                  id="streaming-toggle"
                  checked={streamingEnabled}
                  onChange={() => setStreamingEnabled(!streamingEnabled)}
                  className="sr-only"
                />
                <div className="block bg-gray-200 w-10 h-5 rounded-full"></div>
                <div
                  className={`absolute left-1 top-1 bg-white w-3 h-3 rounded-full transition-transform ${
                    streamingEnabled ? 'transform translate-x-5' : ''
                  }`}
                ></div>
              </div>
            </div>
          </div>
          
          {llmStatus && (
            <div className="flex items-center">
              <span 
                className={`inline-block w-3 h-3 rounded-full mr-2 ${
                  llmStatus.is_processing ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                title={llmStatus.is_processing ? 'LLM is busy' : 'LLM is ready'}
              ></span>
              <button 
                onClick={() => setShowLlmStatus(!showLlmStatus)} 
                className="text-xs text-gray-500 hover:text-gray-700"
                type="button"
              >
                {llmStatus.is_processing ? 'LLM Processing...' : 'LLM Ready'}
              </button>
            </div>
          )}
        </div>
      </form>
      
      {/* LLM Status Panel */}
      {showLlmStatus && llmStatus && (
        <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded-md text-xs">
          <h3 className="font-medium text-gray-700 mb-2">LLM Status:</h3>
          <div className="space-y-1 text-gray-600">
            <p>Model: {llmStatus.last_model_used || 'None used yet'}</p>
            <p>Total queries: {llmStatus.total_queries}</p>
            <p>Successful queries: {llmStatus.successful_queries}</p>
            <p>Cache entries: {llmStatus.cache_size}</p>
            {llmStatus.last_query_time && (
              <p>Last query: {formatTime(llmStatus.last_query_time)}</p>
            )}
          </div>
        </div>
      )}
        {queryStatus && (
        <div className={`mt-4 p-3 rounded-md ${
          queryStatus.state === 'searching' || queryStatus.state === 'generating'
            ? 'bg-blue-50 border border-blue-200 text-blue-700'
            : queryStatus.state === 'success'
              ? 'bg-green-50 border border-green-200 text-green-700'
              : 'bg-yellow-50 border border-yellow-200 text-yellow-800'
        }`}>
          <div className="flex items-center">
            {(queryStatus.state === 'searching' || queryStatus.state === 'generating') && (
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            )}
            <div>
              <div>{queryStatus.message}</div>
              {queryStatus.retrievalTime && (
                <div className="text-xs mt-1">Retrieved documents in {queryStatus.retrievalTime}ms</div>
              )}
              {queryStatus.queryTime && (
                <div className="text-xs mt-1">Total query time: {queryStatus.queryTime}ms</div>
              )}
              {streamingEnabled && queryStatus.state === 'generating' && (
                <div className="text-xs mt-1 font-semibold">Streaming response...</div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-md">
          {error}
          {error.includes("ensure Ollama is running") && (
            <div className="mt-2 text-sm">
              <p>To fix this issue:</p>
              <ol className="list-decimal pl-5 mt-1 space-y-1">
                <li>Make sure Ollama is installed on your computer</li>
                <li>Run <code className="bg-gray-100 px-1 rounded">ollama run llama3.2:latest</code> in a terminal</li> 
                <li>Or set an OpenAI API key in your backend environment</li>
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryInput;