import React, { useState } from 'react';

const ApiKeyModal = ({ 
  isOpen, 
  onClose, 
  provider, 
  providerDisplayName,
  onSubmit,
  loading = false 
}) => {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');

  const getProviderInfo = () => {
    switch (provider) {
      case 'openai':
        return {
          placeholder: 'sk-...',
          description: 'Enter your OpenAI API key to access GPT models',
          helpText: 'Your API key should start with "sk-" and can be found in your OpenAI dashboard.',
          link: 'https://platform.openai.com/api-keys',
          linkText: 'Get OpenAI API Key',
          required: true
        };
      case 'gemini':
        return {
          placeholder: 'AIza...',
          description: 'Enter your Google Gemini API key to access Gemini models',
          helpText: 'Your API key can be generated in Google AI Studio.',
          link: 'https://aistudio.google.com/app/apikey',
          linkText: 'Get Gemini API Key',
          required: true
        };
      case 'huggingface':
        return {
          placeholder: 'hf_...',
          description: 'Enter your Hugging Face token (optional for local models)',
          helpText: 'A token enables access to more models and faster inference. Local models work without it.',
          link: 'https://huggingface.co/settings/tokens',
          linkText: 'Get Hugging Face Token',
          required: false
        };
      default:
        return {
          placeholder: 'Enter API key...',
          description: 'Enter your API key',
          helpText: '',
          link: '',
          linkText: '',
          required: true
        };
    }
  };

  const providerInfo = getProviderInfo();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Validate API key format
    if (providerInfo.required && !apiKey.trim()) {
      setError('API key is required for this provider');
      return;
    }

    if (provider === 'openai' && apiKey && !apiKey.startsWith('sk-')) {
      setError('OpenAI API keys should start with "sk-"');
      return;
    }

    if (provider === 'gemini' && apiKey && !apiKey.startsWith('AIza')) {
      setError('Google Gemini API keys typically start with "AIza"');
      return;
    }

    if (provider === 'huggingface' && apiKey && !apiKey.startsWith('hf_')) {
      setError('Hugging Face tokens should start with "hf_"');
      return;
    }

    try {
      await onSubmit(apiKey.trim());
      setApiKey('');
      onClose();
    } catch (err) {
      setError(err.message || 'Failed to configure API key');
    }
  };

  const handleCancel = () => {
    setApiKey('');
    setError('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full">
        {/* Header */}
        <div className="bg-blue-50 border-b border-blue-200 px-6 py-4 rounded-t-lg">
          <h3 className="text-lg font-semibold text-blue-900">
            Configure API Key
          </h3>
          <p className="text-sm text-blue-700 mt-1">
            {providerDisplayName}
          </p>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="p-6">
          <div className="mb-4">
            <p className="text-sm text-gray-700 mb-3">
              {providerInfo.description}
            </p>
            
            {providerInfo.helpText && (
              <p className="text-xs text-gray-500 mb-3">
                {providerInfo.helpText}
              </p>
            )}

            <label htmlFor="api-key-input" className="block text-sm font-medium text-gray-700 mb-2">
              API Key {providerInfo.required && <span className="text-red-500">*</span>}
            </label>
            
            <input
              id="api-key-input"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={providerInfo.placeholder}
              disabled={loading}
              className={`
                w-full px-3 py-2 border rounded-md text-sm
                focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                ${loading ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}
                ${error ? 'border-red-300' : 'border-gray-300'}
              `}
              autoFocus
            />

            {error && (
              <p className="text-red-600 text-xs mt-1">{error}</p>
            )}
          </div>

          {providerInfo.link && (
            <div className="mb-4 p-3 bg-gray-50 rounded-md">
              <p className="text-xs text-gray-600 mb-2">Don&apos;t have an API key?</p>
              <a 
                href={providerInfo.link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 text-xs underline"
              >
                {providerInfo.linkText} →
              </a>
            </div>
          )}

          <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3 mb-4">
            <div className="flex items-start space-x-2">
              <svg className="h-4 w-4 text-yellow-600 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <div className="text-xs text-yellow-800">
                <strong>Security Notice:</strong> Your API key is temporarily stored in your browser session and sent directly to the model provider. It is not saved permanently or shared with our servers.
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={handleCancel}
              disabled={loading}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || (providerInfo.required && !apiKey.trim())}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Configuring...
                </span>
              ) : (
                'Configure API Key'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ApiKeyModal;
