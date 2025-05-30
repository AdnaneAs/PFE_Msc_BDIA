import React from 'react';

const SearchConfigurationSection = ({
  config,
  onSearchStrategyChange,
  onMaxSourcesChange,
  onQueryDecompositionToggle,
  disabled
}) => {
  const { query_decomposition, search_strategy, max_sources } = config;

  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Search Configuration</h3>
        <p className="text-sm text-gray-600 mt-1">
          Configure search behavior and document retrieval settings
        </p>
      </div>

      <div className="p-6 space-y-6">
        {/* Query Decomposition */}
        <div>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700">
                Enable Query Decomposition (Beta)
              </label>
              <p className="text-sm text-gray-600 mt-1">
                {query_decomposition.description}
              </p>
            </div>
            <button
              onClick={() => !disabled && onQueryDecompositionToggle(!query_decomposition.enabled)}
              disabled={disabled}
              className={`
                relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                ${query_decomposition.enabled ? 'bg-blue-600' : 'bg-gray-200'}
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              <span
                className={`
                  inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                  ${query_decomposition.enabled ? 'translate-x-6' : 'translate-x-1'}
                `}
              />
            </button>
          </div>
          {query_decomposition.enabled && (
            <div className="mt-3 p-3 bg-blue-50 rounded-md">
              <div className="text-sm text-blue-800">
                ✓ Query decomposition is enabled
              </div>
              <div className="text-xs text-blue-600 mt-1">
                Complex questions will be automatically broken down for better results
              </div>
            </div>
          )}
        </div>

        {/* Search Strategy */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Search Strategy
          </label>
          <div className="space-y-3">
            {Object.values(search_strategy.options).map((strategy) => (
              <button
                key={strategy.name}
                onClick={() => !disabled && onSearchStrategyChange(strategy.name)}
                disabled={disabled}
                className={`
                  w-full p-4 rounded-lg border-2 text-left transition-all duration-200
                  ${search_strategy.current === strategy.name
                    ? 'border-blue-500 bg-blue-50 text-blue-900'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                  }
                  ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                `}
              >
                <div className="flex items-center">
                  <div className={`
                    w-4 h-4 rounded-full border-2 mr-3
                    ${search_strategy.current === strategy.name
                      ? 'border-blue-500 bg-blue-500'
                      : 'border-gray-300'
                    }
                  `}>
                    {search_strategy.current === strategy.name && (
                      <div className="w-full h-full rounded-full bg-white scale-50"></div>
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-sm">{strategy.display_name}</div>
                    <div className="text-xs text-gray-600 mt-1">{strategy.description}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>
          <div className="mt-3 p-3 bg-green-50 rounded-md">
            <div className="text-sm text-green-800">
              <strong>Current strategy:</strong> {search_strategy.options[search_strategy.current].display_name}
            </div>
          </div>
        </div>

        {/* Max Sources */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Max Sources
          </label>
          <p className="text-sm text-gray-600 mb-4">
            {max_sources.description}
          </p>
          
          <div className="flex flex-wrap gap-2">
            {max_sources.options.map((option) => (
              <button
                key={option}
                onClick={() => !disabled && onMaxSourcesChange(option)}
                disabled={disabled}
                className={`
                  px-4 py-2 rounded-md border-2 transition-all duration-200 text-sm font-medium
                  ${max_sources.current === option
                    ? 'border-blue-500 bg-blue-500 text-white'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                  }
                  ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                `}
              >
                {option} Sources
              </button>
            ))}
          </div>
          
          <div className="mt-3 p-3 bg-gray-50 rounded-md">
            <div className="text-sm text-gray-700">
              <strong>Currently using:</strong> {max_sources.current} sources per query
            </div>
            <div className="text-xs text-gray-500 mt-1">
              More sources provide comprehensive answers but may increase response time
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchConfigurationSection;
