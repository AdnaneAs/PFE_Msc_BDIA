import React, { useState, useEffect } from 'react';
import { submitQuery, submitDecomposedQuery, getLLMStatus, getSystemConfiguration } from '../services/api';

const QueryInput = ({ onQueryResult }) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queryStatus, setQueryStatus] = useState(null);
  const [llmStatus, setLlmStatus] = useState(null);
  const [showLlmStatus, setShowLlmStatus] = useState(false);

  // Search strategy states - fetched from configuration
  const [searchStrategy, setSearchStrategy] = useState('hybrid');
  const [maxSources, setMaxSources] = useState(5);
  
  // Query decomposition states - fetched from configuration
  const [useDecomposition, setUseDecomposition] = useState(false);
  const [decompositionResult, setDecompositionResult] = useState(null);
  
  // Configuration state
  const [currentConfig, setCurrentConfig] = useState(null);
  
  // Fetch current configuration
  useEffect(() => {
    const fetchConfiguration = async () => {
      try {
        const config = await getSystemConfiguration();
        setCurrentConfig(config);
        
        // Update states with current configuration
        if (config.search_configuration) {
          setSearchStrategy(config.search_configuration.search_strategy.current);
          setMaxSources(config.search_configuration.max_sources.current);
          setUseDecomposition(config.search_configuration.query_decomposition.enabled);
        }
      } catch (err) {
        console.error("Failed to fetch configuration:", err);
      }
    };
    fetchConfiguration();
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
      // Streaming eventSourceRef cleanup removed
    };
  }, []);

  const handleQuestionChange = (e) => {
    setQuestion(e.target.value);
    setError(null);
    setQueryStatus(null);
    setDecompositionResult(null); // Clear previous decomposition results
  };

  const getCurrentModelConfig = () => {
    // Use configuration from backend instead of local state
    if (!currentConfig?.model_selection?.llm) {
      // Fallback to default if no configuration loaded
      return {
        provider: 'ollama',
        model: 'llama3.2:latest'
      };
    }
    
    const { llm } = currentConfig.model_selection;
    const provider = llm.current_provider;
    const model = llm.current_model;
    
    if (provider === 'ollama') {
      return {
        provider: 'ollama',
        model: model
      };
    } else if (provider === 'openai') {
      return {
        provider: 'openai',
        model: model,
        api_key: '' // API keys should be configured separately
      };
    } else if (provider === 'gemini') {
      return {
        provider: 'gemini',
        model: model,
        api_key: '' // API keys should be configured separately
      };
    } else if (provider === 'huggingface') {
      return {
        provider: 'huggingface',
        model: model,
        use_local: true // Default to local usage
      };
    }
    
    // Default fallback
    return {
      provider: 'ollama',
      model: 'llama3.2:latest'
    };
  };

  // Streaming submit handler removed

  const handleSubmitDecomposed = async (e, isRetry = false) => {
    if (e) e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }
    
    if (!isRetry) {
      setLoading(true);
      setError(null);
      setDecompositionResult(null);
    }
    
    // Update status to show decomposition steps
    setQueryStatus({
      state: 'decomposing',
      message: 'Analyzing query complexity...'
    });
    
    try {
      // Get current model configuration
      const modelConfig = getCurrentModelConfig();
      
      // Send the query to the decomposed endpoint
      const result = await submitDecomposedQuery(
        question, 
        modelConfig, 
        searchStrategy, 
        maxSources, 
        useDecomposition
      );
      
      console.log("Decomposed query complete. Result:", result);
      
      // Store decomposition result for display
      setDecompositionResult(result);
      
      // Update status
      setQueryStatus({
        state: 'success',
        message: result.is_decomposed 
          ? `Query decomposed into ${result.sub_queries.length} sub-queries and processed successfully!`
          : 'Query processed as a simple query (no decomposition needed)',
        queryTime: result.total_query_time_ms,
        decompositionTime: result.decomposition_time_ms,
        synthesisTime: result.synthesis_time_ms
      });
      
      if (result) {
        onQueryResult({
          ...result,
          decomposed: true // Flag to indicate this was a decomposed query
        });
      } else {
        onQueryResult({ error: true, message: 'No result returned from backend.' });
      }
      
      // Fetch updated LLM status after query completion
      fetchLlmStatus();
    } catch (err) {
      console.error('Error submitting decomposed query:', err);
      setError(err.message || 'Error submitting decomposed query');
      setQueryStatus({
        state: 'error',
        message: 'Failed to process decomposed query'
      });
      onQueryResult({ error: true, message: err.message || 'Error submitting decomposed query' }); 
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitNormal = async (e, isRetry = false) => {
    if (e) e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
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
      // Get current model configuration
      const modelConfig = getCurrentModelConfig();
      
      // Send the query to the backend
      const result = await submitQuery(question, modelConfig, searchStrategy, maxSources);
      
      // Update status
      setQueryStatus({
        state: 'success',
        message: 'Answer generated successfully!',
        queryTime: result.query_time_ms
      });
      
      console.log("Normal query complete. Result:", result);
      
      if (result) {
        onQueryResult({
          ...result,
          // streaming: false (removed, only non-streaming mode)
        });
      } else {
        onQueryResult({ error: true, message: 'No result returned from backend.' });
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
      onQueryResult({ error: true, message: err.message || 'Error submitting query' }); 
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    if (useDecomposition) {
      handleSubmitDecomposed(e);
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
      
      {/* Current Configuration Summary */}
      {currentConfig && (
        <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-700">Current Configuration</h3>
            <div className="text-xs text-gray-500">Use the settings gear ⚙️ to modify</div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Model:</span>{' '}
              <span className="font-medium text-gray-800">
                {currentConfig.model_selection?.llm?.current_provider || 'ollama'} - {currentConfig.model_selection?.llm?.current_model || 'llama3.2:latest'}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Search:</span>{' '}
              <span className="font-medium text-gray-800">
                {searchStrategy} ({maxSources} sources)
              </span>
            </div>
            <div>
              <span className="text-gray-600">Decomposition:</span>{' '}
              <span className={`font-medium ${useDecomposition ? 'text-blue-600' : 'text-gray-600'}`}>
                {useDecomposition ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>
        </div>
      )}
      
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
              {/* Streaming toggle removed */}
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
          queryStatus.state === 'searching' || queryStatus.state === 'generating' || queryStatus.state === 'decomposing'
            ? 'bg-blue-50 border border-blue-200 text-blue-700'
            : queryStatus.state === 'success'
              ? 'bg-green-50 border border-green-200 text-green-700'
              : 'bg-yellow-50 border border-yellow-200 text-yellow-800'
        }`}>
          <div className="flex items-center">
            {(queryStatus.state === 'searching' || queryStatus.state === 'generating' || queryStatus.state === 'decomposing') && (
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
              {queryStatus.decompositionTime && (
                <div className="text-xs mt-1">Decomposition time: {queryStatus.decompositionTime}ms</div>
              )}
              {queryStatus.synthesisTime && (
                <div className="text-xs mt-1">Synthesis time: {queryStatus.synthesisTime}ms</div>
              )}
              {queryStatus.queryTime && (
                <div className="text-xs mt-1">Total query time: {queryStatus.queryTime}ms</div>
              )}
              {/* Streaming response message removed */}
            </div>
          </div>
        </div>
      )}
      
      {/* Decomposition Results Display */}
      {decompositionResult && decompositionResult.is_decomposed && (
        <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-md">
          <h4 className="text-sm font-medium text-purple-800 mb-2">Query Decomposition Results</h4>
          <div className="space-y-3">
            <div>
              <p className="text-xs text-purple-700 font-medium">Original Query:</p>
              <p className="text-sm text-purple-600 italic">"{decompositionResult.original_query}"</p>
            </div>
            <div>
              <p className="text-xs text-purple-700 font-medium">Sub-queries Generated:</p>
              <ol className="list-decimal list-inside space-y-1 text-sm text-purple-600">
                {decompositionResult.sub_queries.map((subQuery, index) => (
                  <li key={index} className="pl-2">{subQuery}</li>
                ))}
              </ol>
            </div>
            {decompositionResult.sub_results && decompositionResult.sub_results.length > 0 && (
              <div>
                <p className="text-xs text-purple-700 font-medium">Sub-query Results:</p>
                <div className="space-y-2">
                  {decompositionResult.sub_results.map((subResult, index) => (
                    <div key={index} className="bg-white p-2 rounded border border-purple-200">
                      <p className="text-xs font-medium text-purple-800">Q{index + 1}: {subResult.sub_query}</p>
                      <p className="text-xs text-gray-600 mt-1">
                        Sources: {subResult.sources_count} | Time: {subResult.query_time_ms}ms
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-md">
          {error}
          {error.includes("Backend server is not available") && (
            <div className="mt-2 text-sm">
              <p>🔧 <strong>Troubleshooting steps:</strong></p>
              <ol className="list-decimal pl-5 mt-1 space-y-1">
                <li>Check if the backend server is running on <code className="bg-gray-100 px-1 rounded">http://localhost:8000</code></li>
                <li>Ensure Ollama is installed and running with <code className="bg-gray-100 px-1 rounded">ollama run llama3.2:latest</code></li>
                <li>Verify your Ollama installation and model availability</li>
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryInput;