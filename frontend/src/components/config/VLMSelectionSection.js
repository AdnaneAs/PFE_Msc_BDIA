import React, { useState } from 'react';

const VLMSelectionSection = ({
  vlmConfig,
  onVLMProviderChange,
  onVLMModelChange,
  onRefreshModels,
  disabled
}) => {
  const [isLoading, setIsLoading] = useState(false);

  if (!vlmConfig || !vlmConfig.available_providers) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg">
        <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800">VLM Configuration</h3>
          <p className="text-sm text-gray-600 mt-1">Loading vision language model settings...</p>
        </div>
        <div className="p-6">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
            <div className="grid grid-cols-2 gap-4">
              <div className="h-20 bg-gray-200 rounded"></div>
              <div className="h-20 bg-gray-200 rounded"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const handleProviderChange = async (provider) => {
    if (disabled) return;
    
    setIsLoading(true);
    try {
      await onVLMProviderChange(provider);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelChange = async (model) => {
    if (disabled) return;
    
    setIsLoading(true);
    try {
      await onVLMModelChange(model);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg">      <div className="bg-gradient-to-r from-purple-50 to-pink-50 px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-800 flex items-center">
              <span className="mr-2">🖼️</span>
              Vision Language Model (VLM)
            </h3>
            <p className="text-sm text-gray-600 mt-1">
              Configure AI models for image description and multimodal search
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-2 text-sm text-purple-600">
              <span className="bg-purple-100 px-2 py-1 rounded-full text-xs font-medium">
                v0.3 Multimodal
              </span>
            </div>
            {onRefreshModels && (
              <button
                onClick={() => !disabled && onRefreshModels()}
                disabled={disabled}
                className={`
                  flex items-center space-x-2 px-3 py-2 text-sm font-medium rounded-lg border transition-colors
                  ${disabled 
                    ? 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed' 
                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400'
                  }
                `}
                title="Refresh available VLM models"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span>Refresh Models</span>
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">        {/* VLM Provider Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            VLM Provider
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
            {Object.values(vlmConfig.available_providers).map((provider) => (
              <button
                key={provider.name}
                onClick={() => handleProviderChange(provider.name)}
                disabled={disabled || isLoading}
                className={`
                  p-4 rounded-lg border-2 text-left transition-all duration-200
                  ${vlmConfig.current_provider === provider.name
                    ? 'border-purple-500 bg-purple-50 text-purple-900'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                  }
                  ${disabled || isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
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
            ))}
          </div>
        </div>        {/* API Key Configuration for providers that require setup */}
        {(vlmConfig.current_provider === 'openai' || vlmConfig.current_provider === 'gemini' || vlmConfig.current_provider === 'huggingface') && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="flex-1">
                <h4 className="text-sm font-medium text-blue-800 mb-2">
                  API Key Configuration - {vlmConfig.available_providers[vlmConfig.current_provider]?.display_name}
                </h4>
                <p className="text-xs text-blue-700">
                  {vlmConfig.current_provider === 'openai' && 'Required: OpenAI API key to access GPT-4 Vision models.'}
                  {vlmConfig.current_provider === 'gemini' && 'Required: Google Gemini API key to access Gemini Pro Vision models.'}
                  {vlmConfig.current_provider === 'huggingface' && 'Optional: Hugging Face token for enhanced vision model access.'}
                </p>
                <div className="mt-2 text-xs text-blue-600">
                  Note: API keys are configured in the LLM section and shared across text and vision models.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Model Selection for Current Provider */}
        {vlmConfig.current_provider !== 'ollama' && 
         vlmConfig.available_providers[vlmConfig.current_provider]?.models && 
         vlmConfig.available_providers[vlmConfig.current_provider].models.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="block text-sm font-medium text-gray-700">
                {vlmConfig.available_providers[vlmConfig.current_provider]?.display_name} Model
              </label>
            </div>
            <div>
              <select
                value={vlmConfig.current_model}
                onChange={(e) => handleModelChange(e.target.value)}
                disabled={disabled || isLoading}
                className={`
                  w-full md:w-auto min-w-[300px] px-3 py-2 border border-gray-300 rounded-md
                  focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500
                  ${disabled || isLoading ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}
                `}
              >
                {vlmConfig.available_providers[vlmConfig.current_provider].models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-2">
                Selected model: <span className="font-medium">{vlmConfig.current_model}</span>
              </p>
              <p className="text-xs text-green-600 mt-1">
                ✓ {vlmConfig.available_providers[vlmConfig.current_provider].models.length} model(s) available
              </p>
              
              {/* Provider-specific tips */}
              {vlmConfig.current_provider === 'openai' && (
                <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-xs text-blue-700">
                    💡 <strong>Tip:</strong> GPT-4o offers the best vision performance. GPT-4-vision-preview is a cost-effective alternative for simpler image analysis.
                  </p>
                </div>
              )}
              
              {vlmConfig.current_provider === 'gemini' && (
                <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-xs text-blue-700">
                    💡 <strong>Tip:</strong> Gemini-1.5-pro offers excellent vision capabilities. Gemini-1.5-flash provides faster processing for simpler tasks.
                  </p>
                </div>
              )}

              {vlmConfig.current_provider === 'huggingface' && (
                <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-xs text-blue-700">
                    💡 <strong>Tip:</strong> BLIP models excel at image captioning. GIT models provide better scene understanding.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Ollama VLM Model Selection */}
        {vlmConfig.current_provider === 'ollama' && (
          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="block text-sm font-medium text-gray-700">
                Ollama VLM Model
              </label>
            </div>
            
            {vlmConfig.available_providers.ollama.models && vlmConfig.available_providers.ollama.models.length > 0 ? (
              <div>
                <select
                  value={vlmConfig.current_model}
                  onChange={(e) => handleModelChange(e.target.value)}
                  disabled={disabled || isLoading}
                  className={`
                    w-full md:w-auto min-w-[300px] px-3 py-2 border border-gray-300 rounded-md
                    focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500
                    ${disabled || isLoading ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}
                  `}
                >
                  {vlmConfig.available_providers.ollama.models.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-2">
                  Selected model: <span className="font-medium">{vlmConfig.current_model}</span>
                </p>
                <p className="text-xs text-green-600 mt-1">
                  ✓ {vlmConfig.available_providers.ollama.models.length} VLM model(s) available locally
                </p>
              </div>
            ) : (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-yellow-600" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text-yellow-800 mb-2">
                      No Ollama VLM Models Available
                    </h4>
                    <p className="text-sm text-yellow-700 mb-3">
                      Either Ollama is not running or no VLM models are installed. To use Ollama for image analysis:
                    </p>
                    <ol className="text-sm text-yellow-700 list-decimal list-inside space-y-1">
                      <li>Start Ollama: <code className="bg-yellow-100 px-1 rounded">ollama serve</code></li>
                      <li>Pull a VLM model: <code className="bg-yellow-100 px-1 rounded">ollama pull llava:latest</code></li>
                      <li>Refresh this page</li>
                    </ol>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Model Configuration Info */}
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <span className="text-purple-500 text-xl">ℹ️</span>
            </div>
            <div className="ml-3">
              <h4 className="text-sm font-medium text-purple-800 mb-1">
                Multimodal Search Capabilities
              </h4>
              <div className="text-sm text-purple-700 space-y-1">
                <p className="flex items-center">
                  <span className="mr-2">✅</span>
                  Image description generation from uploaded documents
                </p>
                <p className="flex items-center">
                  <span className="mr-2">✅</span>
                  Combined text + image search results
                </p>
                <p className="flex items-center">
                  <span className="mr-2">✅</span>
                  Enhanced responses with visual context
                </p>
              </div>
              <div className="mt-2 text-xs text-purple-600">
                Current provider: <span className="font-medium">{vlmConfig.current_provider}</span> | 
                Model: <span className="font-medium">{vlmConfig.current_model}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VLMSelectionSection;
