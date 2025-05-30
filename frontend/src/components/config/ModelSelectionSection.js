import React, { useState } from 'react';
import ApiKeyModal from './ApiKeyModal';

const ModelSelectionSection = ({
  config,
  onEmbeddingModelChange,
  onLLMProviderChange,
  onLLMModelChange,
  onApiKeyChange,
  apiKeys = {},
  disabled
}) => {  const { llm, embedding } = config;
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKeyLoading, setApiKeyLoading] = useState(false);
  
  const handleApiKeyChange = (provider, value) => {
    // Notify parent component of API key changes
    if (onApiKeyChange) {
      onApiKeyChange(provider, value);
    }
  };

  const handleProviderChange = (provider) => {
    // If the provider requires setup and no API key is configured, show the modal
    const providerConfig = llm.available_providers[provider];
    const needsApiKey = providerConfig?.status === 'unavailable' || 
                       (provider === 'openai' || provider === 'gemini' || provider === 'huggingface');
    const hasApiKey = apiKeys[provider] && apiKeys[provider] !== '';

    if (needsApiKey && !hasApiKey) {
      // First change the provider
      onLLMProviderChange(provider);
      // Then show the API key modal
      setShowApiKeyModal(true);
    } else {
      // Provider doesn't need API key or already has one
      onLLMProviderChange(provider);
    }
  };

  const handleApiKeySubmit = async (apiKeyValue) => {
    setApiKeyLoading(true);
    try {
      await handleApiKeyChange(llm.current_provider, apiKeyValue);
    } finally {
      setApiKeyLoading(false);
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Model Selection</h3>
        <p className="text-sm text-gray-600 mt-1">
          Configure AI models for language processing and embeddings
        </p>
      </div>

      <div className="p-6 space-y-6">
        {/* LLM Provider Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            LLM Provider
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
            {Object.values(llm.available_providers).map((provider) => (              <button
                key={provider.name}
                onClick={() => !disabled && handleProviderChange(provider.name)}
                disabled={disabled}
                className={`
                  p-4 rounded-lg border-2 text-left transition-all duration-200
                  ${llm.current_provider === provider.name
                    ? 'border-blue-500 bg-blue-50 text-blue-900'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                  }
                  ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                  ${provider.status === 'unavailable' ? 'opacity-60' : ''}
                `}
              >
                <div className="font-medium text-sm">{provider.display_name}</div>
                <div className="text-xs text-gray-500 mt-1">{provider.description}</div>
                <div className={`text-xs mt-2 font-medium ${
                  provider.status === 'available' ? 'text-green-600' : 'text-orange-600'
                }`}>
                  {provider.status === 'available' ? '● Available' : '● Requires Setup'}
                </div>
              </button>
            ))}          </div>
        </div>

        {/* API Key Configuration for providers that require setup */}
        {(llm.current_provider === 'openai' || llm.current_provider === 'gemini' || llm.current_provider === 'huggingface') && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="flex-1">
                <h4 className="text-sm font-medium text-blue-800 mb-2">
                  API Key Configuration - {llm.available_providers[llm.current_provider]?.display_name}
                </h4>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`flex items-center space-x-2 text-sm ${
                      (apiKeys[llm.current_provider] && apiKeys[llm.current_provider] !== '') ? 'text-green-600' : 'text-orange-600'
                    }`}>
                      <span className={`w-3 h-3 rounded-full ${
                        (apiKeys[llm.current_provider] && apiKeys[llm.current_provider] !== '') ? 'bg-green-500' : 'bg-orange-500'
                      }`}></span>
                      <span className="font-medium">
                        {(apiKeys[llm.current_provider] && apiKeys[llm.current_provider] !== '') ? 'API Key Configured' : 'API Key Required'}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    {(apiKeys[llm.current_provider] && apiKeys[llm.current_provider] !== '') ? (
                      <>
                        <button
                          onClick={() => setShowApiKeyModal(true)}
                          disabled={disabled}
                          className="px-3 py-1 text-xs font-medium text-blue-700 bg-blue-100 hover:bg-blue-200 rounded-md disabled:opacity-50"
                        >
                          Update Key
                        </button>
                        <button
                          onClick={() => handleApiKeyChange(llm.current_provider, '')}
                          disabled={disabled}
                          className="px-3 py-1 text-xs font-medium text-red-700 bg-red-100 hover:bg-red-200 rounded-md disabled:opacity-50"
                        >
                          Clear Key
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={() => setShowApiKeyModal(true)}
                        disabled={disabled}
                        className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Configure API Key
                      </button>
                    )}
                  </div>
                </div>
                
                <p className="text-xs text-blue-700 mt-2">
                  {llm.current_provider === 'openai' && 'Required: OpenAI API key to access GPT models.'}
                  {llm.current_provider === 'gemini' && 'Required: Google Gemini API key to access Gemini models.'}
                  {llm.current_provider === 'huggingface' && 'Optional: Hugging Face token for enhanced model access.'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Ollama Model Selection */}
        {llm.current_provider === 'ollama' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Ollama Model
            </label>
            <select
              value={llm.current_model}
              onChange={(e) => !disabled && onLLMModelChange(e.target.value)}
              disabled={disabled}
              className={`
                w-full md:w-auto min-w-[200px] px-3 py-2 border border-gray-300 rounded-md
                focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                ${disabled ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}
              `}
            >
              {llm.available_providers.ollama.models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-2">
              Selected model: <span className="font-medium">{llm.current_model}</span>
            </p>
          </div>
        )}

        {/* Embedding Model Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Embedding Model Selection
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.values(embedding.available_models).map((model) => (
              <button
                key={model.name}
                onClick={() => !disabled && onEmbeddingModelChange(model.name)}
                disabled={disabled}
                className={`
                  p-4 rounded-lg border-2 text-left transition-all duration-200
                  ${embedding.current_model.name === model.name
                    ? 'border-blue-500 bg-blue-50 text-blue-900'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                  }
                  ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                `}
              >
                <div className="font-medium text-sm mb-2">{model.display_name}</div>
                <div className="text-xs text-gray-600 mb-3">{model.description}</div>
                
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500">Dimensions:</span>
                    <span className="ml-1 font-medium">{model.dimensions}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Size:</span>
                    <span className="ml-1 font-medium">{model.model_size}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Speed:</span>
                    <span className={`ml-1 font-medium ${
                      model.speed === 'Fast' ? 'text-green-600' : 'text-orange-600'
                    }`}>
                      {model.speed}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Quality:</span>
                    <span className={`ml-1 font-medium ${
                      model.quality === 'Excellent' ? 'text-blue-600' : 'text-green-600'
                    }`}>
                      {model.quality}
                    </span>
                  </div>
                </div>
                
                <div className="mt-2 text-xs text-gray-500">
                  <strong>Use case:</strong> {model.use_case}
                </div>
              </button>
            ))}
          </div>
          
          {embedding.current_model && (
            <div className="mt-3 p-3 bg-blue-50 rounded-md">
              <div className="text-sm text-blue-800">
                <strong>Currently using:</strong> {embedding.current_model.display_name}
              </div>
              <div className="text-xs text-blue-600 mt-1">
                {embedding.current_model.dimensions} dimensions • {embedding.current_model.model_size} • {embedding.current_model.speed} speed
              </div>
            </div>
          )}        </div>
      </div>
      
      {/* API Key Configuration Modal */}
      <ApiKeyModal
        isOpen={showApiKeyModal}
        onClose={() => setShowApiKeyModal(false)}
        provider={llm.current_provider}
        providerDisplayName={llm.available_providers[llm.current_provider]?.display_name}
        onSubmit={handleApiKeySubmit}
        loading={apiKeyLoading}
      />
    </div>
  );
};

export default ModelSelectionSection;
